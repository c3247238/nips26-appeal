# Critique: Background and Related Work

## Summary Assessment
The related work section is well-organized into four logical subsections that map directly to the paper's contributions. It covers SAE architectures, absorption definition and measurements, mitigation approaches, and the confound problem. The narrative builds momentum toward the paper's contributions, particularly the identification of the L0 confound gap. However, several claims lack citations, some literature coverage has gaps relevant to the paper's positioning, and the transition from Section 2.3 to 2.4 introduces a framing shift that weakens the logical flow.

## Score: 7/10
**Justification**: Solid coverage of the core literature with a clear narrative arc. Reaching 8 requires fixing the unsupported claims in Section 2.4, tightening cross-section consistency with the intro and method, closing the citation gaps on mitigation approaches, and removing the few banned-pattern constructions that survive. The confound-problem subsection (2.4) is the strongest part; the SAE architecture survey (2.1) is competent but slightly formulaic.

## Critical Issues

### Issue 1: DeepMind safety team claim lacks citation
- **Location**: Section 2.4, paragraph 3
- **Quote**: "DeepMind's safety research team publicly deprioritized SAE research in 2025 after finding that dense linear probes dramatically outperform 1-sparse SAE probes on harmful intent detection."
- **Problem**: This is a strong factual claim about an organization's research priorities and specific findings. No citation is provided. "Their report identifies feature absorption as a key culprit" references a document that is never cited. If this refers to an internal blog post, white paper, or public statement, it must be cited. If the source is informal (Twitter/X, oral communication), it should be marked as such or removed.
- **Fix**: Add the specific citation (URL, report title, author list, date). If the claim cannot be sourced to a citable document, either remove the paragraph or rewrite it as: "There is growing concern in the safety community that dense linear probes outperform sparse SAE probes for harmful intent detection, raising questions about whether absorption contributes to this gap [citation needed]."

### Issue 2: "SynthSAEBench, 2026" citation is unverifiable
- **Location**: Section 2.2, paragraph 3
- **Quote**: "a limitation explicitly noted by Chanin et al. and at least four subsequent papers (Karvonen et al., 2025; Korznikov et al., 2025; Li et al., 2025; SynthSAEBench, 2026)"
- **Problem**: "SynthSAEBench, 2026" is not a standard citation format (no author list). The intro section lists five papers noting this limitation (Chanin et al., 2024; Karvonen et al., 2025; Korznikov et al., 2025; Bussmann et al., 2025; Li et al., 2025) -- this is a different and more credible set. The related work section uses a different subset with a non-standard reference. This inconsistency undermines trust in the citation accuracy.
- **Fix**: Replace "SynthSAEBench, 2026" with the proper author-year citation. Align the citation list with the intro's list (which includes Bussmann et al., 2025 instead of SynthSAEBench).

### Issue 3: Unified theoretical framework cited only by arXiv ID
- **Location**: Section 2.3, paragraph 5
- **Quote**: "A unified theoretical framework (arXiv:2512.05534) casts all sparse dictionary learning methods as a piecewise biconvex optimization problem"
- **Problem**: Citing only an arXiv number is insufficient for a venue submission. The reader cannot evaluate the source without navigating away. The paper has authors and a title.
- **Fix**: Replace "A unified theoretical framework (arXiv:2512.05534)" with "Author et al. (2025) cast all sparse dictionary learning methods..." using proper author-year format with the full reference in the bibliography.

## Major Issues

### Issue 4: Inconsistent SAE count between related work and method/intro
- **Location**: Section 2.4, paragraph 1
- **Quote**: "Chanin et al. (2024) report that absorption correlates with downstream quality ($r = -0.595$ across 54 Gemma Scope SAEs)"
- **Problem**: The intro states "48 Gemma Scope SAEs" and the method (Section 3.1.1) explains that 6 canonical SAEs lacking L0 are excluded, yielding 48. Section 2.2 also references "54 Gemma Scope SAEs." The related work should clarify that 54 is the original Chanin et al. dataset, while the paper's analyses use 48. Currently, the reader encounters "54" in Section 2, then "48" in the intro and method, creating confusion about whether these are the same or different datasets.
- **Fix**: Add a parenthetical in Section 2.4 when first mentioning the 54-SAE dataset: "(54 Gemma Scope SAEs; our Phase 1 analysis uses 48 after excluding 6 without reported $L_0$)" or defer the 54 number entirely to Section 2.2 and use "Chanin et al.'s Gemma Scope SAE dataset" in 2.4.

### Issue 5: Section 2.3 last paragraph makes a forward-reference claim without sufficient grounding
- **Location**: Section 2.3, final paragraph
- **Quote**: "Our scaling surface analysis (Section 3.3) provides complementary evidence by mapping which regions of the (width, $L_0$) hyperparameter space exhibit high absorption, independent of architecture choice."
- **Problem**: The phrase "independent of architecture choice" overstates what the scaling surface delivers. The method section (3.3.1) notes 360/420 SAEs are standard (L1), 54 are JumpReLU, 6 are unknown. The discussion (5.4) explicitly acknowledges "architecture subgroup analysis has limited power." Claiming architecture-independence here is misleading.
- **Fix**: Rewrite as: "Our scaling surface analysis (Section 3.3) provides complementary evidence by mapping which regions of the (width, $L_0$) hyperparameter space exhibit high absorption across the 420 SAEs in the SAEBench collection."

### Issue 6: Tian et al. (2025) coverage is shallow relative to its importance
- **Location**: Section 2.2, paragraph 3
- **Quote**: "Tian et al. (2025) frame absorption as a special case of poor feature *sensitivity*..."
- **Problem**: This is the only work that reframes absorption within a broader evaluation framework (sensitivity), which directly relates to this paper's cross-domain measurement attempt. The paper is dismissed in two sentences. Key questions a reviewer would ask: What is their scalable evaluation methodology? How does their sensitivity metric compare to the Chanin dominance-based metric? Does their finding about poor recall anticipate the "super-absorber" phenomenon found in this paper's Phase 2?
- **Fix**: Expand to 3-4 sentences covering their evaluation method, scale (how many SAEs/features), and how their sensitivity framing connects to this paper's finding that the dominance metric measures feature concentration rather than probe-direction-specific absorption.

### Issue 7: Missing discussion of Narayanaswamy et al. (2026) limitations
- **Location**: Section 2.3, paragraph 4
- **Quote**: "Narayanaswamy et al. (2026) proposed masked regularization, which randomly masks high-frequency tokens during SAE training to disrupt the co-occurrence patterns that enable absorption."
- **Problem**: This is the only 2026 reference. It gets two sentences with no quantitative absorption reduction numbers, unlike every other mitigation method in the subsection (Matryoshka: "best absorption scores," OrtSAE: "65% reduction," ATM: "0.0068"). The asymmetric treatment is noticeable.
- **Fix**: Either add the quantitative absorption reduction from Narayanaswamy et al. or note that they do not report standard absorption metrics, explaining the gap.

## Minor Issues

- **Section 2.1, sentence 1**: "Sparse autoencoders decompose polysemantic LLM activations into an overcomplete sparse basis of approximately monosemantic directions." The intro uses nearly identical wording: "Sparse autoencoders (SAEs) decompose polysemantic neural network activations into overcomplete sparse bases of interpretable latents." One of these should be cut or differentiated. The related work version should assume the reader already has the intro's definition and start with the technical detail ($\mathbf{x} \in \mathbb{R}^d$, etc.) directly.
- **Section 2.1, paragraph 2**: "The architecture family has diversified rapidly." This is a banned-pattern-adjacent generic transition. Replace with a concrete statement, e.g., "Five SAE architectures have emerged since 2023, each addressing a different training limitation."
- **Section 2.2, paragraph 1**: The description of the Chanin et al. measurement protocol is dense but essential. Consider breaking the single long sentence ("Their measurement protocol trains linear probes...applies integrated-gradients ablation...") into a numbered list for readability, matching the numbered-step format used in the method section (3.2.3).
- **Section 2.2**: "On the first-letter spelling task across mid-layer Gemma Scope SAEs, Chanin et al. report absorption rates of 15--35%." The intro says "across hundreds of Gemma Scope SAEs." The related work says "mid-layer Gemma Scope SAEs." Clarify which is the correct scope.
- **Section 2.3, paragraph 1**: "Chanin et al., 2025" appears as a reference for feature hedging at Matryoshka inner levels. In Section 2.4, "Chanin and Garriga-Alonso (2025)" discusses feature hedging. Are these the same paper? The citation format inconsistency (et al. vs. named co-author) suggests they may be different works. Clarify.
- **Section 2.4, final sentence**: "Our Phase 2 (Section 3.2) tests cross-domain generalizability on knowledge hierarchies." The actual Phase 2 result (0% cosine-calibrated absorption, metric failure) is a negative/null finding. The forward reference here implies a positive test of generalizability, which could mislead readers about what to expect.
- **Notation**: The section uses $\tau_{\text{cos}} = 0.025$ and $\tau_{\text{dom}} = 1.0$ as the Chanin et al. thresholds, consistent with notation.md. Verified.
- **Glossary compliance**: "dictionary widths" (Section 2.1) is correct per glossary ("dictionary width," not "size" or "capacity"). "TopK" and "JumpReLU" capitalization matches glossary. "SAEBench" is correct (one word). "Gemma Scope" is two words, consistent.

## Visual Element Assessment
- [x] Figures/tables match outline plan (outline specifies no figures for Section 2; the section includes none)
- [x] All visuals referenced before appearance (N/A -- no visuals in this section)
- [x] Captions are self-explanatory (N/A)
- [x] No text-heavy sections that need visual support (the mitigation approaches in 2.3 could benefit from a comparison table summarizing method / mechanism / absorption reduction / limitations, but this is optional given space constraints)

## What Works Well
1. **Section 2.4 (The Unresolved Confound Problem)** is the standout. It builds the confound argument with specific numbers ($r = -0.595$, the width/L0 distribution of high- vs. low-absorption SAEs, the missing L0 control), names the exact gap (no formal causal inference methods), and connects to the paper's Phase 1 contribution. The paragraph beginning "The confound is sharpened by Chanin and Garriga-Alonso (2025)" effectively distinguishes hedging from absorption, setting up a nuanced distinction that the method section exploits.
2. **The mitigation survey (Section 2.3)** covers five distinct approaches with quantitative results for each, and the final paragraph correctly identifies that the mechanisms differ -- nesting, orthogonality, temporal masking, token masking, anchoring -- without attempting a premature unification. This avoids overreach.
3. **Section 2.2's formalization** of the Chanin et al. protocol (probe training, k-sparse probing, false-negative identification, integrated-gradients ablation, dual-threshold detection) is precise enough that a reader could replicate the pipeline from this description alone, which is valuable given that the method section adapts this protocol for cross-domain measurement.
