# Critique: Method

## Summary Assessment
The Method section presents a well-structured, replicable protocol with clear subsections that map directly to the outline. The SAE selection, hierarchy construction, and absorption measurement are described with sufficient specificity for reproduction. However, several critical and major issues undermine the section's rigor: a misleading Random-SAE description, a missing architecture (OrtSAE) that was promised in the proposal, an unreported discrepancy in the absorption formula, and a lack of justification for the k=10 choice.

## Score: 6/10
**Justification**: The section is organized and mostly complete, but the Random-SAE description contains a factual error (permutes W_dec, not W_enc), the proposal's promised OrtSAE is absent without explanation, and the formula discrepancy between the method text and the actual SAEBench implementation (max of three terms vs. max of two ratios) is unaddressed. These issues would raise red flags for a careful reviewer. To reach 7+, fix the Random-SAE description, explain the missing OrtSAE, and clarify the formula. To reach 8+, add justification for k=10 and discuss why all semantic hierarchies achieve perfect resid_acc = 1.0.

---

## Critical Issues

### Issue 1: Random-SAE Description is Factually Incorrect
- **Location**: Section 2.1, paragraph 2
- **Quote**: "The Random-SAE control permutes the decoder matrix $W_{\text{dec}}$ of the Standard SAE, destroying any learned structure while preserving the marginal activation statistics."
- **Problem**: The actual implementation permutes the **encoder** matrix $W_{\text{enc}}$, not the decoder matrix $W_{\text{dec}}$. This is a critical error because permuting $W_{\text{enc}}$ vs. $W_{\text{dec}}$ have different theoretical implications. Permuting $W_{\text{enc}}$ means the SAE still reconstructs using the original decoder directions but with scrambled feature detection; permuting $W_{\text{dec}}$ would mean the features are detected normally but reconstructed with scrambled directions. The code uses `W_enc_permuted = W_enc[:, perm]` (column permutation on the encoder).
- **Fix**: Change "decoder matrix $W_{\text{dec}}$" to "encoder matrix $W_{\text{enc}}$" throughout. Also update the notation.md if it contains the same error.

### Issue 2: Absorption Formula in Text Does Not Match the Actual Computation
- **Location**: Section 2.4, equation for $A_{\text{full}}$
- **Quote**: "$A_{\text{full}} = \max\left(0, \frac{\text{acc}_{\text{resid}} - \text{acc}_{\text{sae}}}{\text{acc}_{\text{resid}}}, \frac{\text{acc}_{\text{resid}} - \text{acc}_{\text{k-sparse}}}{\text{acc}_{\text{resid}}}\right)$"
- **Problem**: The actual SAEBench implementation computes $\max(0, \text{ratio}_1, \text{ratio}_2)$ where the ratios are already the relative accuracy drops. But looking at the raw data, every single hierarchy shows $\text{resid\_acc} = 1.0$ and $\text{sae\_acc} = 1.0$ for all SAEs on semantic hierarchies. This means the first ratio $(\text{acc}_{\text{resid}} - \text{acc}_{\text{sae}}) / \text{acc}_{\text{resid}}$ is always **zero**. The absorption score is therefore driven entirely by the k-sparse term. This is a massive structural feature of the data that the Method section completely hides. The section should explicitly note that $\text{acc}_{\text{sae}} = \text{acc}_{\text{resid}}$ for all semantic hierarchies, making the absorption formula reduce to a single term.
- **Fix**: Add a sentence after the formula: "In practice, for all semantic hierarchies in our suite, $\text{acc}_{\text{sae}} = \text{acc}_{\text{resid}} = 1.0$, so the absorption score simplifies to $(\text{acc}_{\text{resid}} - \text{acc}_{\text{k-sparse}}) / \text{acc}_{\text{resid}}$." This transparency is essential for readers to understand what the metric is actually measuring.

---

## Major Issues

### Issue 3: Missing OrtSAE Architecture Without Explanation
- **Location**: Section 2.1, Table 1
- **Problem**: The research proposal (proposal.md, Section "SAE Selection") explicitly lists "OrtSAE (low absorption)" as one of the architectures to evaluate. Table 1 in the Method section includes 8 architectures but OrtSAE is absent, replaced by PAnneal (not mentioned in the proposal's SAE list) and Random-SAE. There is no explanation for this substitution. A reviewer will notice the discrepancy and question whether the omission was principled or opportunistic.
- **Fix**: Add a footnote or sentence in Section 2.1 explaining why OrtSAE was excluded (e.g., "OrtSAE was excluded because no public checkpoint was available for Pythia-160M layer 8 at the time of evaluation; PAnneal was substituted as the lowest-absorption alternative from the SAELens release.")

### Issue 4: No Justification for k=10 in K-Sparse Probing
- **Location**: Section 2.4, paragraph 3
- **Quote**: "The probe is trained on the top-$k$ sparse latent vector $z^{(k)}$, where only the $k = 10$ largest entries are retained and all others are zeroed."
- **Problem**: The choice of k=10 is presented without justification. With m=2048 latent dimensions, k=10 represents only 0.5% of features. Why 10 and not 5, 20, or 50? The SAEBench first-letter evaluator uses a feature-splitting threshold $\tau_{\text{fs}}$ to dynamically determine k per sample, not a fixed k. The semantic-hierarchy adaptation uses a fixed k=10, which is a methodological deviation that should be justified.
- **Fix**: Add justification: "We set $k = 10$ following the SAEBench default for first-letter evaluation at $\tau_{\text{fs}} = 0.03$, which typically retains 8-12 features per sample. A fixed $k$ simplifies comparison across hierarchies with varying activation sparsity." Alternatively, if this was an arbitrary choice, acknowledge it as a limitation.

### Issue 5: Perfect resid_acc = 1.0 and sae_acc = 1.0 for All Hierarchies is Unusual and Unexplained
- **Location**: Section 2.4, implicit in data
- **Problem**: The raw JSON data shows that for every single hierarchy and every single SAE, both $\text{resid\_acc} = 1.0$ and $\text{sae\_acc} = 1.0$. This means the probes achieve perfect classification on both base-model activations and full SAE latents. While the Method section mentions "All hierarchies achieved perfect probe AUROC ($1.0$)" in Section 2.3, it does not discuss the implications: (1) the task is trivially easy for the probes, and (2) the absorption metric is entirely driven by k-sparse degradation, not by any loss in the full SAE representation. This is a crucial methodological point that affects interpretation.
- **Fix**: Add a paragraph in Section 2.4 noting that probes achieve perfect accuracy on both residual and full-SAE representations for all hierarchies, meaning the absorption score reflects only the information loss from sparsification (k-sparse probing), not from the SAE encoding itself. This has implications for whether the metric measures "absorption" (information loss in the full representation) or merely "sparse recoverability" (whether the top-k features suffice).

### Issue 6: Sentence Template Description is Vague
- **Location**: Section 2.3, paragraph 2
- **Quote**: "For each concept, we generate $N = 100$ sentences using simple templates (e.g., 'The {concept} is a place with rich history.')."
- **Problem**: The template example is given for "building" but the actual templates used for all concepts are not documented. The word "simple" is vague. What other templates were used? Were they concept-specific or generic? This matters for reproducibility and for understanding whether template choice introduces bias (e.g., syntactic similarity between parent and child templates could inflate or deflate absorption).
- **Fix**: Either list all templates in an appendix table (as promised in outline.md Table 4 plan, though that table is for per-hierarchy scores) or describe the template generation strategy more precisely: "Templates were generated by filling a small set of generic frames (e.g., 'The {concept} is...', 'A {concept} can be...') with concept-appropriate predicates. See Appendix A.5 for the full template list."

---

## Minor Issues

- **Section 2.1, Table 1**: The "Expected Absorption" column is labeled as subjective expectations, but no source or justification is given for these assignments. Add a footnote: "Expected absorption levels are based on reported scores in the SAELens documentation and SAEBench leaderboard."

- **Section 2.2**: The primary metric is stated as `mean_full_absorption_score`, but the actual results use `official_absorption_full` from the SAEBench evaluator. Clarify whether these are the same or if there is a distinction.

- **Section 2.3, Table 2**: All Probe AUROC values are 1.000. This is suspiciously perfect. While the text acknowledges this, it does not explain why every hierarchy achieves perfection. Is this because the synthetic sentences are too stereotyped? A brief discussion would strengthen credibility.

- **Section 2.4, paragraph 1**: "We replicate the exact SAEBench absorption logic on our WordNet hierarchies" -- but Section 2.2 already says the first-letter absorption uses the "official SAEBench evaluator." The semantic-hierarchy absorption does NOT use the official evaluator; it uses a custom reimplementation. This should be clarified: "We reimplement the SAEBench absorption logic for our custom hierarchies, matching the official evaluator's formula and hyperparameters."

- **Section 2.5**: The non-hierarchy pairs include "tree -- wood" (Material) and "stone -- rock" (Synonym), but "tree" also appears as a parent in the hierarchy condition (tree -> ash, poon). This creates a mild overlap between conditions. While not fatal, it should be noted: "Note that 'tree' appears in both the hierarchy condition (as a parent) and the non-hierarchy condition (paired with 'wood'); we verified that this overlap does not affect results (see Appendix)."

- **Section 2.6, H1**: "Supported if $r > 0.6$ and the CI excludes $0$; rejected if the CI includes values $< 0.3$" -- these criteria create an ambiguous middle zone ($0.3 \leq r \leq 0.6$ or CI spanning both sides) that is not labeled. The results fall in this zone, so the "inconclusive" label is appropriate, but the criteria themselves should be more clearly defined.

- **Notation inconsistency**: The method uses $\bar{A}_{\text{SH}}$ and $\bar{A}_{\text{NH}}$ but the notation.md also defines $A_{\text{random}}$ which is never used in the Method section. Not a problem, but check if it's used elsewhere.

- **Missing figure reference**: The Method section mentions "Figure 1: fig_method_pipeline_desc.md" in the comment block but this figure is never referenced in the body text. Either reference it or remove it from the plan.

- **Terminology**: Section 2.1 uses "MatryoshkaBatchTopK" but the glossary abbreviates to "Matryoshka." Be consistent: either use the full name from the table or note the abbreviation.

---

## Visual Element Assessment
- [x] Figures/tables match outline plan -- Tables 1, 2, 3 are all present and match the outline. Figure 1 (pipeline diagram) is mentioned in comments but not in body text.
- [x] All visuals referenced before appearance -- Tables 2 and 3 are inline and referenced appropriately. Table 1 is also inline.
- [x] Captions are self-explanatory -- Table captions are minimal but sufficient given the body text.
- [ ] No text-heavy sections that need visual support -- Section 2.4 (Absorption Measurement Protocol) is text-heavy and would benefit from a small diagram showing the three probe levels (residual -> SAE latent -> k-sparse) with arrows indicating information flow. The outline's Figure 1 (pipeline diagram) would address this but is not referenced in the body.

---

## What Works Well
1. **Clear structure mapping to outline**: Each subsection (2.1-2.6) directly corresponds to an outline bullet, making the section easy to navigate and verify for completeness. The transitions between subsections are smooth.

2. **Specific hyperparameters**: The Method provides exact values for nearly every parameter (layer 8, d=768, m=2048, N=100, 200 Adam epochs, lr=0.05, k=10, tau_fs values, B=10,000). This level of specificity supports reproducibility.

3. **Explicit control conditions**: The inclusion of both a non-hierarchy control (Section 2.5) and a Random-SAE control (Section 2.1) demonstrates thoughtful experimental design, even if the Random-SAE description contains an error. The pre-registered hypothesis criteria (Section 2.6) add rigor.
