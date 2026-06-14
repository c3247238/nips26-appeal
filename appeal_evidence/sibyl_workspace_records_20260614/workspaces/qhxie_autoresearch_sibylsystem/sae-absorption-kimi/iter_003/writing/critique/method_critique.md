# Critique: Method

## Summary Assessment
The Method section presents a clear, well-structured protocol that is largely reproducible. The three-probe measurement protocol, WordNet hierarchy construction, and statistical hypotheses are all precisely specified. However, the section contains a critical inconsistency between the stated H1 threshold and the actual results reported elsewhere, a missing justification for the k=10 choice, and an incomplete description of the Random-SAE control construction that could impede replication.

## Score: 6/10
**Justification**: The section is technically competent and well-organized, but the H1 threshold inconsistency (stated as r > 0.6 in Method, but results use "CI excludes 0" as the actual decision rule) is a significant cross-section inconsistency. The missing k=10 justification and incomplete Random-SAE description are gaps that a careful reviewer would flag. To reach 7-8, fix the threshold inconsistency, add the k=10 rationale, and clarify the Random-SAE construction.

## Critical Issues

### Issue 1: H1 Threshold Inconsistency Between Method and Results
- **Location**: Section 2.6, H1 paragraph
- **Quote**: "H1 is supported if r > 0.6 and the CI excludes 0."
- **Problem**: The Method section states two conditions for H1 support: (1) r > 0.6 and (2) CI excludes 0. But the Results section (3.2) reports r = 0.463 with CI [-0.389, 0.981] and concludes "the evidence neither supports nor rejects H1." If the decision rule were truly "r > 0.6 AND CI excludes 0," then H1 would be *rejected* (not "inconclusive") because r = 0.463 < 0.6. The Results section instead treats the CI width as the deciding factor, effectively using "CI excludes 0" as the sole criterion. This is a logical inconsistency between sections.
- **Fix**: Either (a) change the Method to state the actual decision rule used in Results (e.g., "H1 is supported if the point estimate r > 0.6 and the 95% CI lies entirely above 0.3"), or (b) change Results to apply the stated rule consistently. The proposal.md uses r > 0.6 as the sole threshold, which is cleaner. Recommend aligning both sections with the proposal: "H1 is supported if r > 0.6 with a 95% CI that excludes values below 0.3."

### Issue 2: Missing Justification for k=10 in K-Sparse Probing
- **Location**: Section 2.4, bullet 3
- **Quote**: "K-sparse probe (acc_k-sparse): Logistic regression trained on the top-k most active SAE latents, where k = 10."
- **Problem**: The choice of k = 10 is arbitrary and unexplained. With m = 2048 latents, k = 10 retains only 0.5% of the latent dimensions. Why not k = 5, k = 20, or a fraction-based threshold? The SAEBench default uses tau_fs = 0.03 (which would retain ~61 latents for m = 2048), not a fixed k. The mismatch between the fixed k = 10 here and the tau_fs-based thresholds in H3 is confusing.
- **Fix**: Add a sentence justifying k = 10. If it follows SAEBench convention, cite it. If it was chosen empirically, state the criterion (e.g., "k = 10 was chosen to match the median number of active latents observed at tau_fs = 0.03 across architectures"). Alternatively, clarify whether k = 10 is used for the main analysis while tau_fs is used only for the robustness analysis in H3.

## Major Issues

### Issue 3: Incomplete Random-SAE Control Description
- **Location**: Section 2.1, paragraph 1
- **Quote**: "The eighth is a Random-SAE control constructed by permuting the decoder directions of the Standard SAE while retaining its trained encoder."
- **Problem**: The description is incomplete for replication. What exactly is permuted---rows or columns of W_dec? Is the permutation random or fixed (seeded)? Does the permuted decoder still satisfy the unit-norm constraint typically enforced in SAE training? The Results section (3.5) and Discussion (4.3) treat this control as central to the paper's main claim, so its construction must be fully specified.
- **Fix**: Expand to: "The Random-SAE control is constructed by applying a fixed random permutation (seed = 42) to the columns of the Standard SAE's decoder matrix W_dec \in \mathbb{R}^{d \times m}, while retaining the encoder W_enc and biases unchanged. The permutation destroys the learned correspondence between latent directions and semantic features while preserving the geometric properties of the decoder basis (column norms and pairwise angles)."

### Issue 4: Absorption Formula Notation Mismatch
- **Location**: Section 2.4, equation
- **Quote**: A_full = max(0, (acc_resid - acc_sae) / acc_resid, (acc_resid - acc_k-sparse) / acc_resid)
- **Problem**: The notation uses "acc" (accuracy) but the text describes probes trained with logistic regression, which output probabilities. The SAEBench protocol (and the notation.md) uses AUROC as the probe quality metric, not accuracy. The Results section reports absorption scores computed from these probes, but it is unclear whether "acc" means classification accuracy, AUROC, or something else. If accuracy is used, the minimum threshold is stated as AUROC >= 0.7---a mismatch.
- **Fix**: Clarify whether "acc" in the formula refers to accuracy or AUROC. If AUROC is used (which is standard in SAEBench), replace "acc" with "AUROC" throughout the formula and text. If accuracy is genuinely used, explain why AUROC is reported for probe quality but accuracy is used for absorption.

### Issue 5: Missing Description of Probe Training Details
- **Location**: Section 2.4
- **Problem**: The probe training procedure is underspecified. What optimizer? How many epochs? What regularization (if any)? How is the train/validation/test split handled? The Results section mentions "3 random seeds for probe training" from the proposal, but the Method does not mention seeds or cross-validation.
- **Fix**: Add a paragraph: "Probes are trained with scikit-learn's LogisticRegression (liblinear solver, L2 regularization, C = 1.0) on 80% of the synthetic data, with 20% held out for evaluation. We report mean AUROC across 3 random seeds for data splitting."

### Issue 6: Non-Hierarchy Control Task Description Is Ambiguous
- **Location**: Section 2.5, paragraph 2
- **Quote**: "the probe classifies which of the two words is present in a sentence, and the accuracy drop from residual to SAE latents quantifies absorption-like behavior."
- **Problem**: For semantic hierarchies, the probe classifies parent vs. child (a binary classification). For non-hierarchy pairs, the task is described as "which of the two words is present"---but this is also binary (word A vs. word B). The description does not explain how this differs structurally from the hierarchy probe. Is the probe trained to detect the presence of either word? Or to distinguish word A from word B? The parallel to the hierarchy task is unclear.
- **Fix**: Clarify: "For non-hierarchy pairs, the probe is trained to distinguish sentences containing word A from sentences containing word B, analogous to the parent-vs-child classification in hierarchies. The same absorption formula is applied, measuring whether the SAE representation loses discriminative information for the pair."

## Minor Issues

- **Section 2.1, line 1**: "Miller, 1995" citation is bare---no parenthetical context. Add "(Miller, 1995)" when WordNet is first mentioned, or use a footnote.
- **Section 2.3, Table 2 caption**: "All hierarchies achieved perfect probe AUROC (= 1.0) on base-model residual activations, validating probe quality." --- This repeats information from the body text ("All probes achieved AUROC = 1.0 on residual activations"). The repetition is minor but unnecessary; the caption should add value, not restate.
- **Section 2.3, line 2**: "NLTK interface" should specify the NLTK function or module used (e.g., "NLTK's WordNet interface, `nltk.corpus.wordnet`").
- **Section 2.6, line 1**: "Three hypotheses guide the statistical analysis" --- "guide" is weak. "frame" or "structure" is more precise.
- **Section 2.6, H3**: "The correlation is stable across feature-splitting thresholds" --- "stable" is vague. "numerically consistent" (as used in Results) is more accurate.
- **Section 2.1**: "activation dimension d = 768 and latent dimension m = 2048" --- these symbols are not defined in the Method section itself. A reader jumping to Method from the Introduction would not know what d and m mean. Add: "where d = 768 is the residual-stream dimension and m = 2048 is the SAE latent dimension."
- **Transition sentence (last line)**: "With the protocol established, we present the empirical findings." --- This is fine but generic. A more specific transition would reference the key test: "With the protocol established, we test whether first-letter absorption predicts semantic-hierarchy absorption."

## Visual Element Assessment
- [x] Figures/tables match outline plan: Table 2 (WordNet hierarchies) and Table 3 (non-hierarchy pairs) are present and match the outline.
- [x] All visuals referenced before appearance: Table 2 is referenced in text ("Table 2 lists the exact hierarchies") before it appears.
- [x] Captions are self-explanatory: Both table captions provide sufficient context.
- [ ] No text-heavy sections that need visual support: Section 2.4 (Absorption Measurement Protocol) describes three probe conditions and a formula. A small diagram showing the probe pipeline (residual -> SAE latents -> k-sparse -> probe) would improve clarity. The outline's Figure & Table Plan does not include a method pipeline diagram, but one would be helpful.

## What Works Well

1. **Clear structure with numbered subsections**: The 2.1-2.6 structure maps cleanly to the outline and makes the protocol easy to follow. Each subsection has a single, well-defined purpose.

2. **Explicit hypothesis specification in 2.6**: The three hypotheses (H1-H3) with their decision criteria are precisely stated. This is strong methodological practice and makes the Results section's assessments traceable---even if the H1 threshold needs alignment.

3. **Transparent control conditions**: The inclusion of both a non-hierarchy control (2.5) and a Random-SAE control (2.1) demonstrates careful experimental design. The explicit listing of all WordNet hierarchies (Table 2) and non-hierarchy pairs (Table 3) supports reproducibility.
