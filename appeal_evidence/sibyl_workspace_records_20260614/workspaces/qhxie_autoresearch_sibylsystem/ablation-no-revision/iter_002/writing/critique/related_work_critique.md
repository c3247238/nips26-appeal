# Critique: Background and Related Work

## Summary Assessment
The section provides a well-organized three-part survey: SAE mechanics and applications (2.1), feature interaction phenomena with a clear absorption-vs-superposition distinction (2.2), and causal analysis with SAEs (2.3), capped by a positioning statement (2.4). The writing is technically precise and the prose is accessible. The section's principal weaknesses are: (1) the positioning statement in 2.4 implicitly promises coverage of token frequency correlation (H2) that was never run; (2) Section 2.3 frames the causal analysis open question without a forward reference to what H4 actually found; and (3) Section 2.4 is redundant with the intro's contribution paragraph and adds no new value for a reader who has already read Section 1.

## Score: 7/10
**Justification**: The section correctly maps the relevant landscape and earns high marks for technical accuracy and the absorption-vs-superposition distinction. It lands at the lower end of the 7 range because: (1) the gap statement in 2.4 is inaccurate — it implies the paper will quantify downstream causal impact, and H2 (token frequency) is never mentioned as untested within the section itself; (2) the causal analysis framing lacks a forward reference to the H4 failure mode; (3) 2.4 is pure repetition of the intro's contribution paragraph. To reach an 8, the section needs either an honest acknowledgment that H2 was not run, or removal of the H2-adjacent framing in 2.3, plus a rewrite of 2.4 that adds value rather than summarizing.

## Critical Issues

### Issue 1: H2 (token frequency correlation) is implicitly promised but never acknowledged as untested
- **Location**: Section 2.3, paragraph 2; Section 2.4 gap statement
- **Quote**: "if some latents redundantly encode the same information, patching those latents would produce correlated effects that complicate causal inference. Conversely, if absorption is rare, most latents are approximately independent..." and "no prior work has systematically quantified how often one latent's feature is redundantly encoded by others — the absorption phenomenon — nor evaluated whether absorption degrades downstream causal analyses"
- **Problem**: The gap statement implies the paper will address both (a) absorption prevalence and (b) downstream causal impact. H1, H3, H4 address these. However, H2 (token frequency correlation) is listed in the intro's Table 1 as a pending hypothesis and is part of the original research questions in `idea/proposal.md` ("does co-occurrence frequency... correlate with absorption rates?"). The method section (Section 4.3) discloses that "H2 was not tested in our pilot due to early termination," but this disclosure is buried in the method — the related work's own framing never signals that H2 was dropped. A reader of the related work section would reasonably expect the paper to address token-frequency correlation effects and whether absorbed features impair circuit tracing. The section sets up these expectations without honestly acknowledging they will not be met.
- **Fix**: In Section 2.3, add one sentence before the faithfulness discussion: "A secondary question — whether low-frequency token features are absorbed at higher rates (H2) — requires corpus-wide token frequency statistics and is addressed in Section 9 pending full-scale experiments." This makes the gap statement in 2.4 accurate about what the paper actually delivers and does not overpromise.

## Major Issues

### Issue 2: Section 2.4 is redundant with the intro's contribution paragraph
- **Location**: Section 2.4, lines 25-30
- **Quote**: "Prior work has established that SAEs produce sparse, interpretable features, that feature representations in neural networks exhibit superposition and other interaction effects, and that activation patching enables causal circuit analysis. However, no prior work has systematically quantified how often one latent's feature is redundantly encoded by others — the absorption phenomenon — nor evaluated whether absorption degrades downstream causal analyses. Our work fills this gap..."
- **Problem**: This paragraph repeats, nearly verbatim, the contribution paragraph already given in the intro (intro lines 13-21). A reader who has read the introduction gets zero new information from Section 2.4. At a top venue, reviewers read the intro and skip to method — this section is therefore wasted. More critically, as noted in Critical Issue 1, the gap statement is inaccurate with respect to H2.
- **Fix**: Replace Section 2.4 with a discussion of open problems beyond the paper's scope: e.g., absorption in larger models (GemmaScope, Llama), absorption under different SAE training objectives, or the relationship between absorption and polysemanticity at scale. This would make the section worth reading for someone who already knows the paper's results and would otherwise skip it.

### Issue 3: Section 2.3 lacks a forward reference to the H4 failure mode and misstates the section number
- **Location**: Section 2.3, paragraph 2, closing
- **Quote**: "if absorption is rare, most latents are approximately independent and SAE-based circuit analysis may be more reliable than feared." and "Our H4 experiments (Section 7) test this directly..."
- **Problem**: This sets up H4 as testing whether absorption level predicts causal importance. The section correctly frames the open question but provides no forward reference to what H4 actually found (both low- and high-absorption subsets yielded 0.0 faithfulness, making the comparison uninformative). Additionally, the paper has no Section 7 — H4 results are in Section 5.3. A reviewer following the cross-reference will hit a dead link.
- **Fix**: Add at the end of 2.3: "We test this directly in Section 5.3 (H4), comparing faithfulness across full SAE patching, low-absorption patching, and high-absorption patching conditions." This corrects the section number and honestly previews the design.

### Issue 4: Section 2.2's empirical claims lack quantification
- **Location**: Section 2.2, lines 11-12 (Sharkey et al. 2023; Templeton 2025)
- **Quote**: "finding that dictionary features often exhibit correlated activation patterns" and "identifying structured clustering in activation space"
- **Problem**: Both findings are described in qualitative language with no specific numbers. "Often exhibit correlated activation patterns" could mean r=0.01 or r=0.8. The glossary cross-check confirms the paper uses precise quantitative language throughout (e.g., "99.4% of latents have $A_f = 0.0$ exactly"). This section's empirical grounding should match that standard.
- **Fix**: Add one specific quantitative finding per citation. For Sharkey et al.: "finding that X% of dictionary feature pairs showed correlation above r=0.3." For Templeton: "identifying structured clustering with a Silhouette score of X at layer Y." If precise numbers are not recoverable from the papers, describe the trend more precisely: "a majority of feature pairs showed positive correlation on the GPT-2 small corpus."

## Minor Issues

- **Line 15, Section 2.2**: "our H5 experiments" should be "our H5 experiments (Section 5.4)" for consistent cross-referencing — currently H5 gets a section parenthetical but the target (Section 8) does not exist in this document. All hypothesis section references need updating: H4 (Section 2.3) → "Section 5.3", H5 (Section 2.2) → "Section 5.4".
- **Line 29, Section 2.4**: "no prior work has systematically quantified" — the word "systematically" is a filler word common in related work sections. "No prior work has quantified..." is more direct.
- **Line 30, Section 2.4**: "our work fills this gap with a reproducible absorption metric" — this repeats the intro's phrasing. Replace with something specific: "We provide a reproducible absorption metric, a systematic multi-hypothesis characterization across layers and dictionary sizes, and the first evaluation of absorption's impact on circuit faithfulness in GPT-2 small."
- **Section 2.1, lines 5-6**: "SAELens ... enables reproducible benchmarking across dictionary sizes and training configurations" — the paper uses one SAE release (`gpt2-small-res-jb`) on one model (GPT-2 small) and does not benchmark across configurations. While the library has this capability, the claim overstates what this paper demonstrates. Change to: "SAELens (Bloomfield et al., 2024) provides a library for loading and evaluating pretrained SAEs, enabling reproducible analysis of SAE feature properties."

- **Glossary cross-check**: The glossary prefers "fine-tuning" over "finetuning." The related_work section does not use "finetuning" — compliant. The intro uses "fine-tuning" correctly.
- **Notation cross-check**: Section 2.3 defines $\text{faithfulness} = \Delta_{\text{patch}} / \Delta_{\text{logit}}$. This matches the notation table exactly — no discrepancy found.

## Visual Element Assessment
- [x] Figures/tables match outline plan (outline specifies no figures for Section 2 — compliant)
- [x] No visuals referenced in text (correct — outline plan: "None")
- [x] No text-heavy passages that need visual support
- [ ] Cross-references to figures in later sections could strengthen the H5 paragraph (e.g., pointing to Figure 2 before the reader reaches Section 8)

## What Works Well

1. **Absorption vs superposition distinction (lines 13-14)**: "Absorption is the more operationally severe failure mode: if feature $f$ is fully absorbed by co-firers $c \in \text{top5}(f)$, then $f$ carries no additional information beyond what is recoverable from those co-firers alone." This is the clearest published articulation of the absorption-superposition distinction this reviewer has encountered. The geometric-vs-variance framing is analytically precise and directly motivates why a reconstruction-based metric (RVE) is necessary rather than a correlation-based one.

2. **SAE quality three-axis framework (lines 7-8)**: "The quality of an SAE is typically assessed along three axes: reconstruction fidelity, sparsity, and interpretability. Notably, existing SAE quality metrics do not measure feature absorption." This is an elegant logical bridge: it names what is measured, then identifies what is not, creating immediate motivation for the paper's contribution without any hyperbolic language.

3. **Faithfulness metric definition (lines 19-21)**: The mathematical definition is correctly stated, the surrounding explanation is clear and self-contained, and the citations to Wang et al. (2022) and Meng et al. (2022) are appropriate. A reader unfamiliar with activation patching could implement it from this description alone.

4. **Logical progression across subsections**: 2.1 (SAE tool) → 2.2 (what can go wrong) → 2.3 (how we measure if it matters) follows a natural pedagogical arc. The subsection headings carry the reader smoothly from tool description to the paper's specific contribution.
