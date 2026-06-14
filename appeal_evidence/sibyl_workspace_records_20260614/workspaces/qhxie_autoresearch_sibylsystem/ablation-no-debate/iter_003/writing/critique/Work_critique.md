# Critique: Background and Related Work

## Summary Assessment
The section covers necessary background on SAEs and absorption concisely, but Section 2.4 ("Factorial Decomposition: Isolating Encoder vs Decoder") crosses the line from literature review into original contribution. This section presents new experimental results (pilot vs full experiment, Condition C variance) that properly belong in the Results section, not Related Work. The conceptual material in 2.1-2.3 is solid, but the section would benefit from deeper engagement with the specific claims in prior work rather than previewing this paper's findings.

## Score: 6/10
**Justification**: A 6-score Related Work section should synthesize prior literature, position the new work within it, and identify gaps—not present new results. Section 2.4 fails this test. Additionally, the outline plan for a conceptual figure (Figure 1) in this section is not reflected in the text, and one citation claim needs verification.

## Critical Issues

### Issue 1: Section 2.4 Presents Original Results, Not Related Work
- **Location**: Section 2.4 ("Factorial Decomposition: Isolating Encoder vs Decoder"), paragraphs 3-4
- **Quote**: "The pilot experiments suggested decoder irrelevance ($C_C \approx C_A$), but the full 5-seed experiment reveals that Condition C has extreme seed-dependent variance (std = 17.13, range 0--43.84), indicating the decoder does contribute in some configurations."
- **Problem**: This is an original experimental result from this paper, not a synthesis or critique of prior work. The factorial experimental design is the paper's primary contribution (Method section content), not related work. Presenting pilot vs full experiment comparisons here breaks the contract of a Related Work section. Reviewers will question why results are previewed before the Methods section.
- **Fix**: Remove the experimental results from Section 2.4. Instead, describe what the factorial design *would* test and what each hypothesis predicts, ending with the identified gap: "A critical gap in prior work is the lack of controlled factorial experiments isolating encoder versus decoder contributions." Do not anticipate the results.

### Issue 2: Citation Claim Lacks Verifiable Source
- **Location**: Section 2.1, paragraph 1
- **Quote**: "with 70% of features judged genuinely monosemantic by human evaluators [@chanin_absorption]"
- **Problem**: This specific quantitative claim (70%) should be verified against the cited source. If it is accurate, it strengthens the section; if not, it is a critical accuracy error. This precise percentage should be traceable.
- **Fix**: Verify the 70% figure against [@chanin_absorption]. If correct, keep it as strong motivating evidence. If approximate or from a different source, correct or remove.

## Major Issues

### Issue 3: Figure 1 from Outline Is Missing
- **Location**: Outline specifies Figure 1 as "conceptual illustration of absorption phenomenon in SAEs" for Section 2 (Background), but the section footer says "<!-- FIGURES: None -->"
- **Problem**: The outline explicitly plans a figure for this section. Section 2.2 ("The Feature Absorption Problem") would be the natural place for a conceptual diagram showing child features substituting for parent features. The absence of a figure creates a text-heavy explanation that is harder to follow.
- **Fix**: Either add the planned conceptual diagram or explicitly note in the section footer why it is omitted. If it appears elsewhere (e.g., in the Introduction), clarify the cross-reference.

### Issue 4: Section 2.5-2.6 Are Thinly Justified as "Related Work"
- **Location**: Sections 2.5 ("Feature Sensitivity and the Pareto Frontier Question") and 2.6 ("Relationship to Hierarchical Feature Structures")
- **Quote**: Section 2.5: "This theoretical prediction motivates the hypothesis (H_Pareto) tested in the Experiments section" — Section 2.6: "The presence of hierarchical structure raises the question..."
- **Problem**: Section 2.5 reads as motivation for this paper's hypothesis, not as review of related work. Section 2.6 mentions Tang et al. and Cunningham et al. but gives only one sentence to each. These feel like transitions or introductions to the paper's contributions rather than substantive literature review.
- **Fix**: Expand 2.5 to review prior work on feature sensitivity (Hu et al.) and the theoretical basis for the Pareto frontier claim. Expand 2.6 to more thoroughly engage with hierarchical feature learning literature. Alternatively, move these to the Introduction or a separate "Theoretical Background" subsection.

### Issue 5: Inconsistent Depth of Prior Work Engagement
- **Location**: Sections 2.1-2.3 vs 2.4-2.6
- **Problem**: Sections 2.1-2.3 give substantive descriptions of prior work (SAE training objective, decoder geometry hypothesis, sparsity optimization hypothesis). Sections 2.4-2.6 pivot to presenting this paper's approach. The section is structurally bipolar: the first half is genuine Related Work; the second half is contribution preview.
- **Fix**: Refocus Section 2.4 on what prior work hypothesized about encoder vs decoder contributions (only the gap analysis), not on what this paper's experiments revealed. Move the experimental results to Results (Section 4).

## Minor Issues

- **Location**: Section 2.1, formula notation. The encoder is described with $W_e \in \mathbb{R}^{n \times d}$ and decoder with $W_d \in \mathbb{R}^{d \times n}$, which is consistent with notation.md. However, the description "encoder network $f(\cdot)$ with weights... that maps input activations to sparse latent representations" should specify whether $f(\cdot)$ includes a sparsity-inducing activation (e.g., TopK). Given the method uses TopK ($k=32$), this detail matters for technical accuracy.
- **Location**: Section 2.3. "Both hypotheses share a common assumption: that the decoder plays a primary role in absorption." — This is a reasonable synthesis, but could be strengthened by citing specific passages from Chanin et al. and Korznikov et al. that support this characterization.
- **Location**: Section 2.6, last paragraph. "Understanding absorption therefore requires understanding how hierarchical structure interacts with the SAE training dynamics." — This sentence is a conclusion/motivation statement, not related work synthesis. It belongs in the Introduction.

## Visual Element Assessment
- [ ] **Figures/tables match outline plan**: NO — Outline plans Figure 1 (conceptual absorption diagram) for this section, but section footer says "None"
- [ ] **All visuals referenced before appearance**: N/A for this section (no visuals present)
- [ ] **Captions are self-explanatory**: N/A
- [ ] **No text-heavy sections that need visual support**: PARTIAL — Section 2.2 would benefit from the planned conceptual figure; text explanation of absorption without visual aid is less clear than it could be

## What Works Well

1. **Section 2.1 SAE Background (paragraphs 1-2)**: The mathematical formulation of the SAE training objective is precise and well-presented. The formula $L = \|\mathbf{x} - \mathbf{\hat{x}}\|^2 + \lambda \|\mathbf{z}\|_1$ is correctly stated, and the description of the encoder-decoder architecture maps cleanly to notation.md. This is the right level of background detail.

2. **Section 2.2 Absorption Definition (paragraph 1)**: The concrete example of a 3-level feature hierarchy with "political statements" as parent and "policy arguments" / "political commentary" as children is well-chosen. It makes the abstract concept of absorption tangible without requiring prior knowledge.

3. **Section 2.3 Gap Identification (paragraph 4)**: "A critical gap in prior work is the lack of controlled factorial experiments isolating encoder versus decoder contributions." — This is exactly what a Related Work section should do: identify the specific gap this paper fills. The two conditional sentences following ("If absorption were purely decoder-driven... conversely...") crisply articulate the empirical prediction that would distinguish the hypotheses. This paragraph is the high point of the section.
