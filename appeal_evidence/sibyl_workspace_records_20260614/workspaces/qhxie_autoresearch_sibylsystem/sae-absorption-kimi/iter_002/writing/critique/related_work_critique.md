# Critique: Related Work

## Summary Assessment
The Related Work section provides a competent survey of the relevant literature but suffers from structural imbalance, incomplete coverage of key citations, and a positioning subsection (2.6) that overclaims and repeats material better handled elsewhere. The section is front-loaded with detailed summaries of individual papers and back-loaded with thin, generic positioning. Several critical papers are missing, and the claim of being "the first" to test construct validity is not fully supported by the literature survey.

## Score: 5/10
**Justification**: The section covers the core literature (Chanin, SAEBench, architecture advances) competently, but the imbalance between detailed summaries and thin synthesis, combined with missing citations and overclaiming in 2.6, keeps it below a 6. To reach a 7, the section needs: (1) a tighter thematic structure that groups related work by problem rather than by paper, (2) complete citation coverage, (3) removal or substantial rewrite of 2.6, and (4) a more nuanced treatment of the novelty claim.

---

## Critical Issues

### Issue 1: Section 2.6 Overclaims and Redundantly Repeats Contributions
- **Location**: Section 2.6, entire paragraph
- **Quote**: "Our study is the first to test the construct validity of the dominant absorption metric on real semantic hierarchies, using a rigorous statistical protocol with bootstrap confidence intervals, paired t-tests, and a random-SAE control."
- **Problem**: This paragraph is essentially a compressed version of the Introduction's contributions and the Conclusion's summary. It does not belong in Related Work. The novelty claim ("the first to test") is also made in the Introduction (1.4), the Results (3.2), and the Discussion (4.1). Repeating it here adds nothing and makes the section feel padded. More critically, the claim is not fully supported by the literature survey itself---the RAVEL discussion (2.4) notes that RAVEL "evaluates factual disentanglement using country-continent hierarchies, but does not frame its task as an absorption metric." This is a distinction without a difference: RAVEL does test hierarchical feature behavior, and the reader is left to trust the authors' framing rather than seeing a clear gap analysis.
- **Fix**: Remove Section 2.6 entirely. Related Work should end with a brief transition sentence pointing forward to the Method section. The positioning of this work belongs in the Introduction (where it already exists) and the Discussion (where it is developed). If a transition is needed, replace 2.6 with a single sentence: "The following section describes how we adapt the SAEBench protocol to WordNet semantic hierarchies."

### Issue 2: Missing Key Citations in Architecture Advances (2.3)
- **Location**: Section 2.3
- **Quote**: "Matryoshka SAEs (Bussmann et al., 2025)... OrtSAE (Korznikov et al., 2025)... Hierarchical SAEs (Zhan et al., 2026)..."
- **Problem**: The section cites three architecture papers but does not mention Rajamanoharan et al. (2024), whose JumpReLU SAE is one of the evaluated architectures in the Results section (Table 1). This is a conspicuous omission given that JumpReLU is included in the experiments. Additionally, the section does not cite any of the foundational SAE papers (e.g., Bricken et al., 2023; Cunningham et al., 2023) that established the SAE framework within which absorption was later identified. Without this context, a reader unfamiliar with SAEs may not understand why absorption matters.
- **Fix**: Add a brief sentence in 2.3 acknowledging that absorption reduction is evaluated within the broader SAE framework established by Bricken et al. (2023) and Cunningham et al. (2023). Add Rajamanoharan et al. (2024) to the architecture list, noting that JumpReLU also reports absorption metrics.

### Issue 3: Incomplete Construct Validity Literature (2.4)
- **Location**: Section 2.4
- **Quote**: "Rajpurkar et al. (2016) showed that SQuAD reading comprehension scores do not predict performance on adversarially modified passages. Bowman & Dahl (2021) argued that many NLP benchmarks conflate multiple constructs..."
- **Problem**: The construct validity discussion is thin. The two cited papers are about general NLP benchmark validity, not specifically about mechanistic interpretability or SAE evaluation. The section misses: (1) recent work on SAE evaluation validity (e.g., Lieberum et al., 2023, which is cited in 2.5 but not here); (2) the broader ML evaluation validity literature (e.g., Ribeiro et al., 2020 on benchmark brittleness); (3) and specifically, any work that questions whether SAE metrics generalize across tasks or models.
- **Fix**: Expand 2.4 with 1-2 additional sentences citing work on SAE metric validity. If no such work exists, explicitly state: "To our knowledge, no prior work has questioned the construct validity of SAE-specific metrics---a gap this paper addresses." Also add Ribeiro et al. (2020) or similar on benchmark brittleness as broader context.

---

## Major Issues

### Issue 4: Structural Imbalance---Front-Loaded Summaries, Back-Loaded Synthesis
- **Location**: Entire section
- **Problem**: Sections 2.1-2.5 each summarize 1-3 papers in detail, while 2.6 attempts to synthesize everything in a single paragraph. The result is a "laundry list" structure where the reader must wait until the final paragraph to see how the pieces connect. A better structure would group related work by the *problem* each line of work addresses: (a) absorption theory and measurement, (b) architecture improvements, (c) benchmark validity concerns, (d) semantic hierarchies in interpretability.
- **Fix**: Restructure into thematic subsections rather than paper-by-paper summaries. For example:
  - 2.1 Measuring Feature Absorption (merge 2.1 + 2.2)
  - 2.2 Architecture Advances Targeting Absorption (keep 2.3)
  - 2.3 Construct Validity and Benchmark Design (expand 2.4)
  - 2.4 Semantic Hierarchies in Interpretability (keep 2.5, with more depth)
  This would allow each subsection to build a narrative arc: what was done, what gap remains, how our work addresses it.

### Issue 5: WordNet Section (2.5) Underdeveloped
- **Location**: Section 2.5
- **Quote**: "WordNet (Miller, 1995) has been used extensively in NLP and interpretability research as a source of structured semantic knowledge. In the context of SAEs, prior work has used WordNet to evaluate feature quality (Bricken et al., 2023) and to construct concept hierarchies for probing (Lieberum et al., 2023). However, these studies do not measure absorption..."
- **Problem**: The section mentions two uses of WordNet but does not explain *how* Bricken et al. and Lieberum et al. used it, or why their approaches differ from ours. The reader cannot evaluate whether our adaptation is genuinely novel or merely a technical variation. The claim "Our work is the first to adapt the SAEBench absorption formula to WordNet hierarchies" is accurate but undersupported---the section does not explain what Bricken et al. and Lieberum et al. actually did.
- **Fix**: Add 1-2 sentences describing each prior WordNet use. For Bricken et al. (2023): "Bricken et al. used WordNet to validate that SAE features correspond to human-annotated semantic categories, measuring alignment rather than information loss." For Lieberum et al. (2023): "Lieberum et al. constructed concept hierarchies from WordNet to test whether SAE features encode hierarchical structure, but did not apply an absorption-specific metric." This makes the gap explicit.

### Issue 6: SAEBench Section (2.2) Understates Technical Adaptation
- **Location**: Section 2.2
- **Quote**: "The benchmark adapted the metric technically by replacing ablation-based criteria with probe-projection criteria, enabling evaluation at all layers rather than only early layers."
- **Problem**: The description of SAEBench's technical adaptation is vague. "Ablation-based criteria" vs. "probe-projection criteria" are not defined, and the significance of this change is not explained. A reader unfamiliar with the original Chanin protocol will not understand why this matters. The section also does not mention that SAEBench reports absorption scores across architectures, layers, and sparsity levels---information that is relevant to the paper's scope (single layer, single model).
- **Fix**: Expand the technical description in one sentence: "Chanin's original protocol required causal ablations that are only tractable in early layers; SAEBench replaced these with probe-projection criteria that measure information loss without intervention, extending absorption evaluation to deeper layers." Then add a sentence noting the limitation: "However, SAEBench's architectural sweep does not include semantic-hierarchy tasks, leaving the metric's domain generalization untested."

### Issue 7: Chanin Summary (2.1) Misrepresents the Detection Protocol
- **Location**: Section 2.1, paragraph 1
- **Quote**: "Chanin et al. introduced a detection protocol based on ground-truth logistic probes and k-sparse probing, measuring the accuracy drop between residual-stream and SAE-latent classifications."
- **Problem**: The summary conflates two different protocols. Chanin et al.'s original protocol used causal ablations (intervening on specific latents), while SAEBench adapted it to use probe-projection (measuring accuracy drops without ablation). The section attributes the probe-projection approach to Chanin, which is inaccurate. This matters because the paper's own method uses probe-projection (Section 2.4), and the reader needs to understand that this is a SAEBench adaptation, not the original Chanin protocol.
- **Fix**: Revise: "Chanin et al. introduced a detection protocol based on causal ablations and ground-truth logistic probes. SAEBench later adapted this protocol to use probe-projection criteria (Section 2.2), which we follow in our experiments."

---

## Minor Issues

- **Line 1 (heading)**: "# 2. Related Work" uses a numbered heading inconsistent with the LaTeX-style section numbering elsewhere (e.g., "# 1 Introduction" in intro.md has no period). Standardize to "# 2 Related Work" (no period) for consistency.
- **Line 7**: "This work established absorption as a central pathology in SAEs, but its evaluation task---first-letter classification..." The em-dash should be preceded by a space or be a proper LaTeX-style em-dash. In markdown, use `---` with spaces: `task --- first-letter`.
- **Line 11**: "SAEBench (Karvonen et al., 2025)" --- the proposal.md cites "Karvonen" while the intro.md uses "Adam-Karvonen" or similar. Check citation key consistency across sections. (The notation.md uses "Adamkarvonen2025" in the intro's \cite command.)
- **Line 19**: "The question of whether a benchmark measures what it claims to measure---construct validity---has received sustained attention in ML evaluation." This is a good opening for 2.4, but "sustained attention" is vague. Specify: "since 2016" or "in the past decade."
- **Line 23**: "Our work is the first to adapt the SAEBench absorption formula to WordNet hierarchies, creating a direct bridge between the benchmark's measurement protocol and real semantic structure." This sentence belongs in the Introduction or Discussion, not Related Work. It is a contribution claim, not a literature summary.
- **Missing citation for RAVEL**: Vig et al. (2024) is cited but not included in the References section of outline.md. Ensure it appears in the final bibliography.
- **Missing year consistency**: Zhan et al. is cited as 2026 in the proposal and outline, but this may be a preprint or future-dated citation. Verify.

---

## Visual Element Assessment
- [x] Figures/tables match outline plan (outline specifies "None" for Related Work)
- [x] All visuals referenced before appearance (N/A---no visuals)
- [x] Captions are self-explanatory (N/A)
- [ ] No text-heavy sections that need visual support: The section is entirely text. While no figures are planned, a small table comparing the key properties of Chanin, SAEBench, and the current work would improve clarity and reduce the density of prose in 2.1-2.2.

---

## What Works Well

1. **Clear identification of the Chanin gap** (paragraph 2.1, last sentence): The direct quote from Chanin et al. calling for "finding examples of feature absorption unrelated to character identification" is an excellent anchor. It gives the reader a concrete, citable reason why the current work is needed.

2. **Explicit stakes in architecture section** (2.3, last sentence): "If first-letter absorption does not predict semantic-hierarchy absorption, these architectures may have been optimized for the wrong target." This is a strong, specific claim that connects the literature to the paper's motivation. It should be preserved in any rewrite.

3. **Good coverage of the three architecture papers** (2.3): Matryoshka, OrtSAE, and Hierarchical SAEs are all correctly identified as reporting absorption reductions. The brief description of each approach (nested latent spaces, orthogonalization, explicit parent-child modeling) is accurate and informative.
