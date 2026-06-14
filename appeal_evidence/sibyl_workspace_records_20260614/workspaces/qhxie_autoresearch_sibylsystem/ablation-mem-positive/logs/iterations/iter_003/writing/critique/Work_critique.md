# Critique: Related Work

## Summary Assessment
The Related Work section provides solid coverage of SAE interpretability literature, correctly positioning the paper's contribution against prior work. The technical content is accurate with properly cited formulas and specific data points. However, critical credibility issues undermine the section: an unsourced citation (Wilkus et al., 2025), a typo in an author name (Sok rotov), and an abrupt ending that fails to articulate the specific research gap the paper fills. The section's logical flow from SAE basics through absorption to the actionability paradox is effective, but Section 2.6 reads as orphaned content that doesn't adequately bridge to the paper's theoretical framework.

## Score: 6/10
**Justification**: The section covers all necessary background topics with accurate technical descriptions and proper citation of major works (Chanin et al., Basu et al., Karvonen et al.). Critical credibility issues (unsourced citation, author name typo) and structural weaknesses (no research gap articulation, ordering inconsistency) prevent a higher score. Reaching 7-8/10 requires resolving the critical issues and adding explicit gap articulation at the end.

## Critical Issues

### Issue 1: Unsourced Citation (Wilkus et al., 2025)
- **Location**: Section 2.6, first paragraph
- **Quote**: "Attributes with high variance across contexts tend to be more context-sensitive and specialized (Wilkus et al., 2025)."
- **Problem**: "Wilkus et al. (2025)" does not appear in the glossary.md Key References section. The section's references are: Chanin et al. (2024), Basu et al. (2026), Cui et al. (2026), Karvonen et al. (2025), Costa et al. (2025), Pearl (2009). This citation appears to be fabricated or incorrectly attributed. This is a critical credibility issue.
- **Fix**: Either (1) remove the citation and reframe as general prior observation without specific attribution, or (2) verify the correct citation and add it to the glossary Key References. If the paper exists, verify the correct author name spelling.

### Issue 2: Typo in Author Name
- **Location**: Section 2.5, "Matryoshka SAEs (Sok rotov et al., 2025)"
- **Quote**: "Matryoshka SAEs (Sok rotov et al., 2025) use nested dictionaries..."
- **Problem**: "Sok rotov" is a clear typographical error. The outline's Key References shows "Sokrov et al., 2025" as the correct name. This typo undermines professional credibility.
- **Fix**: Change "Sok rotov" to "Sokrov" (verify against actual paper).

## Major Issues

### Issue 3: No Explicit Research Gap Articulation
- **Location**: End of Section 2.6 (final two sentences)
- **Quote**: "We build on this intuition by connecting CV to steering actionability specifically. Our finding that high-CV absorbed features are more steerable than low-CV absorbed features suggests that CV captures something about the causal routing..."
- **Problem**: The section ends by previewing the paper's findings without stating what prior work leaves unresolved. The reader cannot determine what specific gap the paper fills. Compare to intro.md which explicitly states: "The field lacks a predictor that connects absorption measurement to intervention utility" and "No method to predict which absorbed features can actually be steered."
- **Fix**: Add a concluding paragraph to Section 2.6 that explicitly states: (1) prior work treats all absorbed features as uniformly non-steerable, (2) no method exists to predict which absorbed features remain steerable, (3) the connection between coefficient of variation and steering actionability is unexplored. This creates clear motivation for the paper's approach.

### Issue 4: Section Ordering Inconsistent with Outline
- **Location**: Section 2.6 ("Variance and Feature Steerability") follows Section 2.5 ("Architectural Solutions")
- **Problem**: According to the outline, Section 2.6 (variance/steerability) should logically precede or follow Section 3 (Theory Framework) since it provides intuition for CV-based decomposition. Currently Section 2.6 appears orphaned between architectural solutions and the abrupt ending. The outline's transition logic states: "Related Work → Theory: Position variance paradox and subpopulation decomposition against existing frameworks" — but Section 2.6 doesn't effectively position against frameworks; it just ends with a preview.
- **Fix**: Either (1) move Section 2.6 content to precede Section 2.5 with explicit transitional text connecting to theoretical framework, or (2) restructure the ending of Section 2.6 to explicitly reference Section 3's theoretical framework.

### Issue 5: "In Contrast" Redundancy
- **Location**: Section 2.5, first and second paragraphs
- **Quote**: Paragraph 1: "In contrast, our work takes a different approach..." Paragraph 2: "These approaches address absorption at the source by modifying SAE training. In contrast, our work..."
- **Problem**: The phrase "In contrast" is used twice in close proximity within the same section. This redundancy weakens the writing and the second instance is redundant given the first already established the contrast.
- **Fix**: Replace the second instance with alternative phrasing: "Our approach differs by..." or "We take a complementary approach..." or restructure to avoid the phrase entirely.

## Minor Issues

- **Section 2.5, first paragraph**: "These approaches" is a vague demonstrative with no clear referent to the preceding sentence. Consider: "Architectural solutions address absorption at the source by modifying SAE training, rather than preventing absorption, we investigate..."
- **Section 2.6, first sentence**: "The relationship between activation variance and feature behavior has been noted in prior work" is vague and would benefit from a citation even if just to general prior observation.
- **Section 2.6, last paragraph**: Split into two paragraphs — one for prior work observation, one for the paper's specific contribution. The current single paragraph mixes prior intuition with novel findings inappropriately.
- **Terminology consistency**: The paper uses "steerable" in Section 2.6 ("more steerable") but the glossary's Preferred Phrasings table does not list "steerable" as an accepted term (it recommends "steering effect" or "steering potential"). Verify consistency across sections.
- **Section 2.4, CE-Bench citation**: Verify author name "Kusingo et al." is correct against the actual paper.

## Visual Element Assessment
- [x] No figures/tables planned for Related Work (per outline)
- [x] No text-heavy passages requiring visual support
- [N/A] Related Work does not typically require visual elements

## What Works Well

1. **Section 2.2 (Feature Absorption)**: The explanation of the A_j absorption detector formula is mathematically correct and the explanation of why high A_j indicates absorption is exemplary: "as the decoder direction aligns poorly with its encoder output due to child feature interference." The causal chain from parent subsumption to false negatives to interpretability undermining is clearly articulated.

2. **Section 2.3 (Actionability Paradox)**: The framing of Basu et al.'s findings is precise with correct specific numbers (98.2% AUROC, 0% output change). The progression from detection capability to intervention failure is logical and effectively motivates the research question. The phrase "interpretability illusion" is evocative and appropriately critical.

3. **Section 2.5 (Architectural Solutions)**: Good coverage of OrtSAE, Matryoshka SAEs, and MP-SAE with accurate descriptions. The positioning statement "our work takes a different approach: rather than preventing absorption, we investigate how to predict which absorbed features remain steerable" effectively clarifies the paper's complementary relationship to architectural solutions.

4. **Logical flow**: The section progresses from general background (SAEs) to specific problem (absorption pathology) to field challenge (actionability paradox) to prior attempts (architectural solutions) to the paper's specific angle (variance/steerability). This creates a clear narrative arc that leads naturally into the paper's contribution.

## Cross-Reference Consistency Check

| Item | Related Work | Other Sections | Status |
|------|--------------|----------------|--------|
| Basu et al. actionability paradox | 98.2% AUROC, 0% steering | intro.md: same numbers | PASS |
| Variance paradox (CV ~7.33 vs 0.01) | Section 2.6 mentions 733x ratio | method.md Section 3.1: same | PASS |
| A_j formula | Properly cited to Chanin et al. (2024) | method.md: same citation | PASS |
| OrtSAE reference | Sharkey et al., 2025 | Not in glossary Key References | WARNING |
| Wilkus et al. (2025) | Section 2.6 | Not in glossary Key References | FAIL |
| Sokrov (Matryoshka) | "Sok rotov" in Work.md | "Sokrov" in outline | FAIL |
| Terminology: "steerable" | Section 2.6 uses it | glossary prefers "steering effect" | WARNING |

## Required Actions

| Priority | Issue | Action |
|----------|-------|--------|
| CRITICAL | Unsourced Wilkus et al. citation | Remove or verify/cite properly |
| CRITICAL | "Sok rotov" typo | Change to "Sokrov" |
| HIGH | No research gap articulation | Add concluding paragraph explicitly stating gap |
| HIGH | Section ordering inconsistency | Add transitional text or restructure Section 2.6 |
| MEDIUM | "In contrast" redundancy | Replace one instance with alternative phrasing |
| LOW | "These approaches" vague referent | Replace with explicit reference |
| LOW | "steerable" terminology | Verify consistency or update to preferred term |