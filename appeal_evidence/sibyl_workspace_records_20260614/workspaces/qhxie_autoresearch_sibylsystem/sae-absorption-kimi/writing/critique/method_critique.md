# Critique: Method

## Summary Assessment
The Method section presents a clear experimental protocol with well-defined hypotheses and statistical tests, but suffers from critical inconsistencies with the proposal, unsupported claims about data properties, and a concerning mismatch between the proposed research questions and the actual experiments described. The writing is generally competent but contains several instances of vague or imprecise language that would trouble a careful reviewer.

## Score: 5/10
**Justification**: The section earns points for clarity of protocol description and proper statistical test specification, but loses significant ground on claim-evidence alignment (the proposal describes four experiments; the method describes adaptations of a single protocol), cross-section consistency with the proposal (hypothesis labels are scrambled), and a critical unsupported claim about the Random SAE construction. To reach 7/10, the section needs to align its hypothesis labels with the proposal, remove unsupported claims, and accurately describe the scope of experiments. To reach 9/10, it would need to address the deeper issue that the method does not actually execute the four experiments promised in the proposal.

## Critical Issues

### Issue 1: Hypothesis Labels Are Scrambled Relative to the Proposal
- **Location**: Section 3.6, all three hypothesis definitions
- **Quote**: "**H1 (Construct Validity).** First-letter absorption scores correlate positively with semantic-hierarchy absorption scores...", "**H2 (Hierarchy Specificity).** Semantic-hierarchy absorption exceeds non-hierarchy control absorption...", "**H3 (Robustness).** The first-letter vs. semantic-hierarchy correlation is stable across feature-splitting thresholds."
- **Problem**: These H1/H2/H3 labels do NOT match the proposal's hypothesis definitions. In the proposal (idea/proposal.md), H1 is "Metric Decomposition" (random/PCA baselines), H2 is "Utility Disconnect" (correlation with downstream metrics), H3 is "Co-occurrence Confound" (hierarchy vs. non-hierarchy), and H4 is "Goodhart's Law" (random-baseline margins). The method section's H1 corresponds to the proposal's H5 (First-Letter vs. Semantic Correlation), the method's H2 corresponds to the proposal's H3 (Co-occurrence Confound), and the method's H3 is not explicitly defined in the proposal at all. This scrambling would confuse readers who cross-reference the proposal or expect consistency with the paper's own research questions.
- **Fix**: Align hypothesis labels with the proposal. If the paper has pivoted to a narrower scope (only construct validity and hierarchy specificity), state this explicitly in the method introduction and relabel hypotheses to avoid collision with the proposal's numbering. Alternatively, reframe the method to match the proposal's four experiments.

### Issue 2: Unsupported Claim About Random SAE First-Letter Absorption
- **Location**: Section 3.1, paragraph 3
- **Quote**: "This preserves the encoder's activation statistics while destroying the learned decoder structure, yielding a baseline that tests whether absorption scores reflect base-model geometry rather than learned SAE structure."
- **Problem**: The claim that the Random SAE "preserves the encoder's activation statistics" is unsupported. The Random SAE is constructed by permuting the decoder columns, but the encoder is retained from the trained Standard SAE. The encoder's output distribution depends on the input data and the encoder matrix; permuting the decoder does not change the encoder's activation statistics. However, the claim as phrased suggests something more nuanced about the relationship between encoder and decoder that is not explained. More critically, the iter_001 official SAEBench results show Standard_trainer_0 has official_absorption_full = 0.027, while the method section's Table 1 reports Standard first-letter absorption as 0.026. These are consistent. But the method does not explain why the Random SAE's first-letter absorption (0.030 in Table 1) is being compared against these values when the official SAEBench evaluator was not run on the Random SAE.
- **Fix**: Clarify exactly what "preserves the encoder's activation statistics" means, or remove the claim if it cannot be precisely justified. Explain how the Random SAE's first-letter absorption was computed (was the official SAEBench evaluator used, or was it estimated?).

### Issue 3: The Method Does Not Execute the Proposed Experiments
- **Location**: Entire section, compared against idea/proposal.md
- **Problem**: The proposal promises four experiments: (E1) Decomposing the metric with random/PCA/fully-random SAEs, (E2) Correlating absorption with downstream utility, (E3) Co-occurrence confound with high/low co-occurrence non-hierarchies, and (E4) Random-baseline-corrected architecture comparison. The method section describes only a single adapted protocol: running the SAEBench absorption evaluator on semantic hierarchies and non-hierarchy pairs. There is no mention of PCA-basis SAEs, fully random SAEs, downstream utility correlation, or multiple co-occurrence conditions. This is a massive scope reduction that is never acknowledged.
- **Fix**: Either (a) rewrite the method to include all four experiments as proposed, or (b) explicitly state in the method introduction that the paper has narrowed its scope to Experiment 3 (co-occurrence confound) and the construct-validity correlation, and explain why the other experiments were deferred.

## Major Issues

### Issue 4: Missing Probe AUROC Evidence
- **Location**: Section 3.4, last paragraph
- **Quote**: "All 10 hierarchies achieved AUROC = 1.0 on the ground-truth probe across all SAEs, indicating perfect separability of parent and child concepts in the residual stream."
- **Problem**: The claim of AUROC = 1.0 (perfect separability) is extremely strong and should be backed by evidence. The iter_002 artifact files show probe AUROCs for individual letters (not hierarchies), and the values are around 0.99, not exactly 1.0. The claim that ALL hierarchies achieved EXACTLY 1.0 across ALL SAEs strains credulity — even with synthetic data, numerical precision typically prevents exactly 1.0. This looks like a ceiling effect artifact that the section mentions later (Section 5.5) but does not interrogate here.
- **Fix**: Report the actual AUROC values with precision (e.g., 0.9997) rather than rounding to 1.0. Acknowledge that values near 1.0 may indicate a ceiling effect that distorts the absorption formula, and reference the limitation discussion.

### Issue 5: Inconsistent Statistical Test Descriptions
- **Location**: Section 3.6, H1 and H2
- **Quote**: H1: "We estimate a bootstrap 95% confidence interval with B = 10,000 resamples. H1 is supported if r > 0.6 and the CI excludes 0." H2: "H2 is supported if t > 0 and p < 0.05 (one-tailed)."
- **Problem**: The H1 threshold (r > 0.6) is arbitrary and not justified. Why 0.6? The proposal's H5 uses the same threshold but calls it a "construct-validity threshold" without citation. For H2, a one-tailed test is used without justification. Given that the observed direction was opposite to predicted (non-hierarchy > hierarchy), a one-tailed test in the predicted direction actually makes the rejection of H2 *stronger*, but the choice should be justified.
- **Fix**: Cite a source for the r > 0.6 threshold or justify it theoretically. For H2, justify the one-tailed test choice or report two-tailed p-values for transparency.

### Issue 6: GPT-2 Replication Is Underdescribed
- **Location**: Not in Method section; only appears in Results (Section 4.6)
- **Problem**: The GPT-2 replication is mentioned in the Results section but has NO description in the Method. The reader learns about it only after seeing the results. This violates standard scientific writing practice where all experimental conditions must be described before results are presented. The proposal mentions GPT-2 as a "replication control" but the method does not describe which SAEs were used, which layer, or how they were selected.
- **Fix**: Add a subsection (3.7) describing the GPT-2 replication: model specification, SAE sources, layer choice rationale, and any deviations from the main protocol.

### Issue 7: Table References to Non-Existent Tables
- **Location**: Section 3.3 and end of section
- **Quote**: "Table 2 lists the exact hierarchies." and "![Table 2: WordNet Semantic Hierarchies...](figures/table2_hierarchies.pdf)"
- **Problem**: The section references Table 2 and Table 3, but these tables are not actually present in the section text — they are only figure placeholders. The hierarchies ARE listed inline in the text ("building (house, school, library), container (box, bag, cup)..."), so Table 2 is redundant. Table 3 (non-hierarchy pairs) is not listed inline at all — only the pairs are named without any structure that would require a table.
- **Fix**: Either (a) remove the table placeholders and present the information inline only, or (b) create actual tables with meaningful additional columns (e.g., WordNet synset IDs, frequency counts, relationship types) that justify their existence.

## Minor Issues

- **Section 3.1, line 1**: "We evaluate eight SAE checkpoints spanning the absorption-rate spectrum reported in the literature (Table 1)." — Table 1 is in the Results section, not the Method. Method sections should not reference tables that appear later. → Move this reference to the Results section or add a forward-reference disclaimer.

- **Section 3.1, line 5**: "The seven trained architectures are: Standard (L1-sparse ReLU), TopK, BatchTopK..." — The parenthetical "L1-sparse ReLU" is never used again; the section consistently refers to "Standard SAE." → Remove the parenthetical or use it consistently.

- **Section 3.2, line 13**: "The primary metric is the mean full absorption score $\bar{A}_{\text{FL}}$ across all 26 hierarchies." — The notation $\bar{A}_{\text{FL}}$ is introduced here but never used again in the section. H1 uses the notation without the bar. → Standardize notation: either use $\bar{A}_{\text{FL}}$ consistently or remove the bar.

- **Section 3.3, line 23**: "Table 2 lists the exact hierarchies." — The hierarchies are actually listed in the text immediately following this sentence, not in a table. → Change to "The exact hierarchies are:" or create an actual table.

- **Section 3.5, line 45**: "These pairs exhibit semantic association (synonymy, co-occurrence, or thematic relatedness) but lack parent-child containment structure." — "doctor-hospital" is not synonymy, co-occurrence, or thematic relatedness in the same sense as the others; it is a thematic/functional association. The categorization is imprecise. → Either list relationship types per pair or use a more general descriptor like "semantic or associative relatedness."

- **Section 3.6, line 61**: "All statistical tests are conducted in Python 3.12 using SciPy (Virtanen et al., 2020) for parametric tests and custom bootstrap resampling for confidence intervals." — The citation format "(Virtanen et al., 2020)" is parenthetical, but no bibliography is provided in the section. Ensure consistent citation format across all sections.

- **Figure/table placeholder comments at end**: The HTML-style comments ("<!-- FIGURES ... -->") are production artifacts that should not appear in the final draft. → Remove before submission.

## Visual Element Assessment
- [ ] Figures/tables match outline plan — No outline.md exists to check against; cannot verify.
- [ ] All visuals referenced before appearance — Table 1 is referenced in Section 3.1 but appears in Results (Section 4.1). This is a forward reference that should be avoided in Method sections.
- [x] Captions are self-explanatory — The figure captions in the placeholders are reasonably descriptive.
- [ ] No text-heavy sections that need visual support — Section 3.3 (hierarchy list) and 3.5 (non-hierarchy list) are text-heavy and would benefit from actual tables. The inline listing of 10 hierarchies with 2-3 children each is hard to parse.

## What Works Well

1. **Clear protocol description in Section 3.4**: The three-probe protocol (ground-truth, SAE latent, k-sparse) is described with precise mathematical notation and clear sequencing. A reader could replicate this protocol from the description alone.

2. **Explicit hypothesis formalization in Section 3.6**: The hypotheses are stated with specific statistical tests, thresholds, and success criteria. This is good practice that enables precise falsification — even if the labels are scrambled relative to the proposal.

3. **Synthetic dataset justification in Section 3.3**: The fixed-template rationale ("ensures that the target concept appears as a single-token mention, controlling for contextual variation") is a clear and valid experimental-control justification. The trade-off with ecological validity is appropriately acknowledged in the Limitations subsection of the Discussion.
