# Critique: Related Work

## Summary Assessment

The Related Work section is well-organized around three axes (measurement, architecture, evaluation) that mirror the paper's three contributions, making the narrative scaffolding clear. Coverage of the absorption and SAE architecture literature is thorough and up-to-date (through April 2026). The section falters on two fronts: (1) several claims about prior work lack the specificity needed for a top venue, and (2) the ecological competition subsection (2.4) is underdeveloped relative to the conceptual weight the paper places on it.

## Score: 6/10
**Justification**: Strong organizational structure and solid literature coverage earn it above average. Reaching a 7--8 would require: fixing the inconsistency between the intro's contribution framing and the related work's gap statements; adding quantitative citations where currently vague; strengthening Section 2.4 with actual equations or a table mapping ecological terms to SAE terms; and resolving the terminology/citation issues flagged below.

## Critical Issues

### Issue 1: Intro previews three contributions but Related Work defines four gap statements
- **Location**: Sections 2.1--2.4, each ending with a "Gap" paragraph
- **Quote (Section 2.4)**: "No prior work formalizes the ecological competition analogy for SAE absorption or derives a quantitative absorption predictor from it."
- **Problem**: The introduction frames three contributions: (1) LV detector, (2) corpus PMI analysis, and (3) downstream impact test. But the Related Work has four subsections with four gap statements, including Section 2.4 (ecological competition) which overlaps with Section 2.1's gap about probe-free detection. The reader encounters redundancy: is the "gap" in 2.1 (no probe-free predictor) the same as 2.4 (no formalized LV analogy)? The answer is yes, but the structure obscures this.
- **Fix**: Merge Section 2.4 into Section 2.1 as a paragraph. The ecological analogy is the *proposed solution* to the gap identified in 2.1, not a separate research area with its own gap. Alternatively, if Section 2.4 is kept separate, explicitly state that it provides the theoretical basis for addressing the gap in 2.1, creating a forward reference rather than a parallel structure.

### Issue 2: The "Gap" in Section 2.3 misrepresents DeepMind's finding
- **Location**: Section 2.3, paragraph 3
- **Quote**: "DeepMind's safety research team publicly deprioritized SAE research in 2025 after finding that dense linear probes dramatically outperform 1-sparse SAE probes on harmful intent detection. The blog post identifies feature absorption as a key culprit but provides no systematic quantification of the absorption--performance link."
- **Problem**: This is attributed to "a blog post" with no arXiv ID, URL, or formal citation. For a claim this central to the paper's motivation (the entire Component 3 exists because of this finding), the source must be precisely cited. If it is genuinely only a blog post, the paper should note the informal nature of the source and explain why the claim is nonetheless worth testing. As written, it reads like a rumor that the authors are elevating to a research motivation.
- **Fix**: Provide the full citation (author names, title, publication date, URL). If the blog post has been superseded by a formal publication, cite that instead. Add one sentence acknowledging the informal nature of the source if appropriate.

## Major Issues

### Issue 3: Section 2.2 claims about JumpReLU are unsupported by the cited source
- **Location**: Section 2.2, paragraph 1
- **Quote**: "JumpReLU achieves state-of-the-art reconstruction fidelity on Gemma 2 9B but, paradoxically, exhibits the highest absorption rates in SAEBench---consistent with the hypothesis that longer training amplifies the sparsity-gradient dynamics that produce absorption."
- **Problem**: Two claims are packed into one sentence and neither is properly sourced. (a) "highest absorption rates in SAEBench" -- this claim is attributed to the JumpReLU paper (arXiv:2407.14435) but the SAEBench data comes from Karvonen et al. (arXiv:2503.09532). Cite the correct source. (b) "consistent with the hypothesis that longer training amplifies the sparsity-gradient dynamics" -- this hypothesis is not stated in either cited paper. It appears to be the authors' interpretation presented as if it were established.
- **Fix**: Cite SAEBench for the absorption rate claim. Explicitly attribute the "sparsity-gradient dynamics" hypothesis to the authors: "We hypothesize this is consistent with..." or cite the source if one exists.

### Issue 4: Section 2.4 is thin relative to its conceptual importance
- **Location**: Section 2.4 (Ecological Competition Models)
- **Problem**: The paper's title contains "Lotka-Volterra," the entire method section is built on the ecological analogy, and the paper tests a specific LV prediction (sharp threshold at alpha=1). Yet the related work devotes only two short paragraphs to the ecological modeling literature. The LV competitive exclusion principle has a rich literature (Gause 1934, MacArthur 1958, Chesson 2000, modern niche theory). A reader familiar with ecology would expect: (a) the precise mathematical statement of the competitive exclusion principle, (b) conditions under which it holds and fails (e.g., environmental fluctuations, spatial structure), and (c) a discussion of why the SAE training process might or might not satisfy those conditions. The absence of (c) is particularly notable given that the detector *fails*---the discussion section could be strengthened if the related work foreshadowed the conditions that would need to hold.
- **Fix**: Expand Section 2.4 to at least three paragraphs: (1) the formal LV competitive exclusion principle with its mathematical conditions, (2) known conditions under which it fails in ecology, (3) the analogy mapping with explicit acknowledgment of where the analogy is tight (niche overlap maps to co-activation) and where it is loose (SAE training is not a population dynamics process).

### Issue 5: Section 2.5 cites "arXiv:2512.05534" without author names
- **Location**: Section 2.5, final paragraph
- **Quote**: "A unified theoretical framework (arXiv:2512.05534) casts all sparse dictionary learning methods as piecewise biconvex optimization..."
- **Problem**: This is the only citation in the entire section given as a bare arXiv number without author names. Every other citation follows the "Author et al. (arXiv:XXXX.XXXXX)" format. This breaks consistency and makes the reference harder to locate.
- **Fix**: Add author names: "Author et al. (arXiv:2512.05534)."

### Issue 6: Missing citation for Chanin et al. hedging paper
- **Location**: Section 2.5, paragraph 3
- **Quote**: "Feature *hedging* (Chanin, Dulka, and Garriga-Alonso, arXiv:2505.11756)"
- **Problem**: This paper (arXiv:2505.11756, May 2025) is very recent. The section describes its contribution accurately, but does not mention whether it has been peer-reviewed. More importantly, the distinction between hedging and absorption is stated but not backed with evidence. The sentence "Hedging occurs in capacity-limited regimes; absorption occurs in hierarchical feature regimes regardless of capacity" makes a strong claim about the independence of absorption from capacity, but the cited source (Chanin et al.) may not make this exact claim.
- **Fix**: Verify that the capacity-independence claim is supported by the cited source. If it is the authors' interpretation, mark it as such. Consider adding a brief example to make the hedging-vs-absorption distinction concrete for readers unfamiliar with hedging.

### Issue 7: Section 2.1 LessWrong post lacks a proper citation
- **Location**: Section 2.1, paragraph 4
- **Quote**: "A LessWrong post ('Looking for Feature Absorption Automatically') attempted cosine-similarity-based detection..."
- **Problem**: No author, no date, no URL. LessWrong posts are informal sources; if cited at all, they need full metadata (author, date, URL) so the reader can assess credibility. The post is used as a baseline comparison point, making proper attribution important.
- **Fix**: Add author name(s), date, and URL. Format consistently with other informal-source citations in the field (many MI papers cite LessWrong posts with full metadata in footnotes).

## Minor Issues

- **Section 2.1, paragraph 1**: "NeurIPS 2025" is stated in parentheses after the arXiv ID for Chanin et al. Confirm this acceptance; if the paper is still under review or only appeared at a workshop, correct the venue. No other paper in the section has its venue stated alongside the arXiv ID, creating inconsistency.
- **Section 2.2, paragraph 3**: "Korznikov et al. (arXiv:2509.22033)" -- the author name is spelled "Korznikov" here but "Korznikov" again in Section 2.3. Verify correct spelling. Also confirm this is the same author group as "Korznikov et al. (arXiv:2602.14111)" in Section 2.3, or clarify if they are different teams.
- **Section 2.2, paragraph 1**: "Bussmann et al. (arXiv:2503.17547; ICML 2025)" -- the intro spells this "Bussmann" but the outline says "Bussman" (missing one 'n'). Verify the correct spelling and make it consistent across all sections.
- **Section 2.1**: The glossary specifies the preferred term is "competition coefficient alpha_ij" (not "LV coefficient"). The related work section uses both "competition coefficient" and "competition coefficient alpha_ij" which is fine, but the outline (Section 2) says "LV coefficient approach" -- ensure the final paper uses the glossary-preferred term everywhere.
- **Section 2.2**: "ATM SAE" is introduced with the full name "Adaptive Temporal Masking" from the paper by Li et al. but the glossary does not include ATM SAE. Consider adding it.
- **Section 2.5, paragraph 1**: "Elhage et al. (2022)" -- no arXiv ID is provided, unlike every other citation in the section. Add it for consistency.
- **Section 2.5, paragraph 2**: "Chanin et al., arXiv:2409.14507" is cited for feature splitting, but this is the same paper already cited in Section 2.1 for absorption. This is fine, but note that the sentence "Splitting creates redundancy; absorption creates omission" is a pithy formulation that should appear in the glossary or method section too, not just here.

## Visual Element Assessment

- [x] Figures/tables match outline plan (outline specifies "None" for Related Work)
- [x] All visuals referenced before appearance (N/A -- no visuals in this section)
- [x] Captions are self-explanatory (N/A)
- [ ] No text-heavy sections that need visual support -- **Section 2.2 would benefit from a small comparison table** listing the five SAE architectures, their absorption-reduction mechanism, and which component of alpha_ij each reduces. This table is implied by the gap statement ("no single quantity unifies them") and would make the argument more concrete. The outline does not plan this table, but it would strengthen the section.

## What Works Well

1. **The four "Gap" paragraphs at the end of each subsection** provide a clear, explicit statement of what the paper addresses. This structure makes it easy for a reviewer to verify that the paper fills genuine holes in the literature rather than solving manufactured problems. The gap statements are specific and falsifiable.

2. **Section 2.5's distinction between absorption, splitting, and hedging** (paragraphs 1--3) is one of the best-written passages in the section. The taxonomy of failure modes (superposition vs. splitting vs. absorption vs. hedging) with their capacity dependencies and width dependencies is genuinely clarifying. The sentence "Splitting creates redundancy; absorption creates omission" is memorable and precise.

3. **The coverage of very recent work** (Narayanaswamy et al. arXiv:2604.06495 for masked regularization, Li et al. arXiv:2510.08855 for ATM SAE, Chanin et al. arXiv:2505.11756 for hedging) demonstrates thorough literature search. The section is current through April 2026, which is appropriate for a NeurIPS 2026 submission.
