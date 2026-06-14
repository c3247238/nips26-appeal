# Critique: Method (The Local Inhibition Graph Framework + Methodology)

## Summary Assessment

The Method section combines the theoretical framework (Section 3) and experimental methodology (Section 4) into a coherent whole. The LCA-SAE structural correspondence is presented as a theorem with proof, and the six-phase experimental pipeline is logically organized. However, critical issues persist: the notation collision in the LCA equation remains unfixed, the tied-weight assumption still lacks quantification for the untied case, H6-H10 are described as protocols without results, and several arbitrary thresholds lack justification. The section scores lower than it could due to these unresolved problems.

## Score: 6/10

**Justification**: The section has strong mathematical presentation and clear experimental structure, but loses points for (1) unresolved notation collision between LCA activation and SAE input, (2) no quantification of how well the tied-weight approximation holds for the actual untied SAE used, (3) H6-H10 described as validation protocols without any reported results, creating a mismatch with the intro's promises, and (4) multiple arbitrary thresholds (0.10 precision, 20% firing increase, r > 0.3) without theoretical justification. To reach 8/10, fix the notation issue, quantify the tied-weight approximation, and either execute H6-H10 or reframe them as proposed experiments.

## Critical Issues

### Issue 1: Notation Collision in LCA Equation (Unfixed from Prior Critique)
- **Location**: Section 3.1, LCA dynamics equation
- **Quote**: "$\tau \cdot \frac{du}{dt} = -u + (b - G \cdot a), \quad a = T(u)$"
- **Problem**: The variable $a$ is used for LCA activation (output after thresholding) but also for SAE input activation in the forward pass equation "$z = \text{ReLU}(W_{\text{enc}} a + b_{\text{pre}})$". The notation.md defines $a$ as "Input activation to SAE", creating a direct conflict. This is confusing because $a$ in the LCA equation is the *output* activation (after thresholding), while $a$ in the SAE equation is the *input*. The proof then claims "$z \approx a$" which is a type error---$z \in \mathbb{R}^{d_{\text{dict}}}$ and $a \in \mathbb{R}^{d_{\text{model}}}$ are different dimensionalities.
- **Fix**: Use $\tilde{a}$ or $a_{\text{LCA}}$ for LCA activation. Change the proof to say "$z \approx T(u)$ where $T(u)$ is the LCA activation" rather than "$z \approx a$". Update notation.md to define the LCA activation separately from SAE input.

### Issue 2: Tied-Weight Assumption vs. Actual Untied SAE (Still Unresolved)
- **Location**: Section 3.1, Theorem and Proof
- **Quote**: "For an SAE with tied weights ($W_{\text{enc}} = W_{\text{dec}}^T$), the decoder correlation matrix $G = W_{\text{dec}}^T W_{\text{dec}}$ is exactly the inhibition matrix from the LCA framework." / "Even with untied weights... the structural correspondence holds approximately."
- **Problem**: The theorem assumes tied weights, but the experiments use gpt2-small-res-jb, a standard pretrained SAE with untied weights. The text says the correspondence "holds approximately" for untied weights but provides no quantification of this approximation. How approximate is it? Does the approximation degrade predictions? The entire empirical contribution rests on this approximation being valid, yet no evidence is provided.
- **Fix**: Add a sentence with actual data: "For the gpt2-small-res-jb SAE, the Frobenius norm $\|W_{\text{enc}} - W_{\text{dec}}^T\|_F / \|W_{\text{dec}}^T\|_F = X.XX$, indicating that the tied-weight assumption holds approximately." Or compute and report the cosine similarity between corresponding rows of $W_{\text{enc}}$ and $W_{\text{dec}}^T$. Without this, a reviewer will rightly question whether the exact theorem has any bearing on the actual experiments.

### Issue 3: H6-H10 Described as Protocols Without Results (Mismatch with Introduction)
- **Location**: Sections 4.4-4.8 and throughout
- **Quote**: "H6 is supported if precision@20 $\geq$ 0.10" / "H7 is supported if $r(\text{inh}, \text{recall}) < -0.3$ with $p < 0.05$"
- **Problem**: The methodology section describes H6-H10 as predictions with falsification criteria, but the introduction's Key Results Preview (Section 1.5) presents these as already-tested hypotheses with specific predicted values ("precision@20 = X.XX", "parent firing +20%"). The experiments section (Section 5.2) also describes them in protocol form without results. This creates a fundamental confusion: are these experiments completed or planned? The intro promises results; the method and experiments sections deliver only protocols.
- **Fix**: If experiments are not executed, the introduction must be revised to frame H6-H10 as *proposed* validation experiments, not reported results. Replace "Our validation experiments test five hypotheses" with "We propose five validation experiments to test the competitive suppression theory." Remove placeholder "X.XX" values. If experiments are complete, add actual results to Sections 4.4-4.8 and Section 5.2, and change to past tense.

## Major Issues

### Issue 4: Missing Baseline Comparisons (Identity Graph Still Omitted)
- **Location**: Section 4.4 (H6 validation)
- **Quote**: "We test enrichment vs. a random baseline (shuffle latent indices; expected precision@20 $\approx$ 0.004) using a Fisher exact test. We also test against a non-absorbed control."
- **Problem**: The proposal promised three baselines (random graph, non-absorbed control, identity graph), but only two are mentioned. The identity graph baseline (only self-loops) tests whether correlations beyond self-similarity matter. This is an important control because $G_{ii} = \|W_{\text{dec}}[i]\|^2$ is always the largest correlation for latent $i$, and absorption pairs might be predicted by self-correlation alone.
- **Fix**: Add the identity graph baseline: "Identity graph baseline: only self-loops ($i \rightarrow i$); tests whether self-correlation alone explains absorption pairs."

### Issue 5: Homeostatic Rebalancing Equation Sign Ambiguity (Still Unclear)
- **Location**: Section 3.4 / 4.8
- **Quote**: "$z'_i = \max\left(0, \; z_i + \alpha \sum_{j \in N(i)} G_{ij} \cdot z_j\right)$"
- **Problem**: The equation adds $\alpha \cdot G_{ij} \cdot z_j$ but if $G_{ij}$ is negative (anti-correlated features), this subtracts from $z_i$. The text says "boosts parent activations by the inhibition it receives" but the equation's sign depends on $G_{ij}$. For positively correlated features ($G_{ij} > 0$), this adds to $z_i$; for negatively correlated features ($G_{ij} < 0$), it subtracts. Is this intended? The biological analogy of "homeostatic plasticity" suggests boosting suppressed neurons, which would require $|G_{ij}|$ rather than signed $G_{ij}$.
- **Fix**: Clarify the sign convention explicitly. If signed $G_{ij}$ is correct, explain: "When $G_{ij} > 0$ (correlated neighbors), the boost is positive. When $G_{ij} < 0$ (anti-correlated), the correction is negative, which is appropriate because anti-correlated features do not compete for the same variance." If the intent is to always boost, change to $|G_{ij}|$.

### Issue 6: Graph Complexity Claim Still Misleading
- **Location**: Section 3.3
- **Quote**: "Complexity: $O(k \cdot d_{\text{dict}} \cdot d_{\text{model}})$ --- feasible for 24K--1M latents on a single GPU."
- **Problem**: Computing all pairwise correlations $G = W_{\text{dec}}^T W_{\text{dec}}$ is $O(d_{\text{dict}}^2 \cdot d_{\text{model}})$. For $d_{\text{dict}} = 1M$, this is $10^{12} \cdot d_{\text{model}}$ operations---not feasible in 2 seconds. The $O(k \cdot d_{\text{dict}} \cdot d_{\text{model}})$ only applies if correlations are computed on-demand for neighbors, not all pairs. The claim conflates full correlation matrix computation with approximate nearest-neighbor methods.
- **Fix**: Clarify: "Computing the full correlation matrix $G = W_{\text{dec}}^T W_{\text{dec}}$ costs $O(d_{\text{dict}}^2 \cdot d_{\text{model}})$. For GPT-2 Small ($d_{\text{dict}} = 24{,}576$), this takes ~2 seconds on an A100. For larger SAEs, approximate nearest-neighbor methods (e.g., FAISS) reduce complexity to $O(k \cdot d_{\text{dict}} \cdot d_{\text{model}})$."

### Issue 7: Arbitrary Thresholds Lack Justification
- **Location**: Sections 4.4, 4.5, 4.7, 4.8
- **Quote**: "Precision@20 $\geq$ 0.10" / "$r < -0.3$" / "$r > 0.3$" / "parent firing increases by $>20\%$"
- **Problem**: All these thresholds are arbitrary. Why 0.10 and not 0.05 or 0.20? Why $r > 0.3$ and not 0.2 or 0.5? Why 20% firing increase and not 10% or 30%? None have theoretical justification. They appear chosen to be achievable rather than derived from the theory. This undermines the falsification framework---if thresholds are arbitrary, falsification is arbitrary too.
- **Fix**: For precision@20, justify: "We set precision@20 $\geq$ 0.10 as the support threshold because it represents a 25$\times$ enrichment over the random baseline (0.004), which we consider a minimally meaningful signal for practical diagnostic use." For correlation thresholds, replace with standard significance testing: "H7 is supported if $r(\text{inh}, \text{recall})$ is significantly negative ($p < 0.05$) and $r(\text{inh}, \text{precision})$ is non-significant ($p > 0.05$)." For H10, replace the 20% threshold with a statistical test or justify based on practitioner needs.

### Issue 8: Hypothesis Numbering Inconsistency Across Paper
- **Location**: Throughout Sections 3-5
- **Problem**: The hypotheses section (hypotheses.md) defines H1, H1b, H2, H3 (absorption-degradation hypotheses from prior work). The method section introduces H6-H10 (inhibition framework validation). H4 and H5 are never formally defined in the hypotheses section but appear in experiments.md (H5 = precision-recall asymmetry) and the proposal (H4 = layer-dependent structure, H5 = homeostatic rebalancing in the original proposal). The numbering gap (H4, H5 missing from hypotheses.md) and the jump to H6-H10 is confusing.
- **Fix**: Add a clear mapping: "We retain H1-H3 from our prior correlation study (Section 3.2) and introduce H6-H10 as validation hypotheses for the inhibition framework. H4 (EC50 analysis) and H5 (precision-recall asymmetry) from prior work are discussed in Section 5.1." Or renumber H6-H10 as H4'-H8' to avoid confusion.

## Minor Issues

- **Section 3.1, proof**: "$z \approx a$" --- $z \in \mathbb{R}^{d_{\text{dict}}}$ and $a \in \mathbb{R}^{d_{\text{model}}}$ are different dimensions. Should say "$z \approx T(u)$".
- **Section 3.3**: "For GPT-2 Small... computing all correlations takes approximately 2 seconds on an A100" --- This timing claim should be verified or labeled as an estimate.
- **Section 4.2**: "Layer 0 provides a near-input baseline; layers 4, 8, and 10 sample the mid-to-late network" --- The layer selection justification is weak. Why these specific layers? Add a citation to prior work or explain the selection criteria.
- **Section 4.5**: The inh_in equation uses $G_{j, \phi(f)}$ but the text says "incoming inhibition" which by the notation convention should be $G_{\phi(f), j}$ (inhibition from j to phi(f)). Check consistency with notation.md definition of $G_{ij}$ as "inhibition from latent j to latent i".
- **Section 4.9**: "Code and evaluation protocol are released with the paper" --- If no repository exists yet, say "will be released upon publication" or provide a placeholder URL.
- **Figure references**: Figure 1 is referenced before its caption description (the image path appears before the caption text). Standard practice is to describe the figure in text first, then show it.
- **Section 4.1 table**: The phase table lists H6-H10 but these hypotheses are never formally stated in the hypotheses section (hypotheses.md only has H1-H3).

## Visual Element Assessment

- [x] Figures/tables match outline plan --- Figure 1 (LCA correspondence) and Figure 2 (suppression mechanism) are planned and referenced
- [ ] All visuals referenced before appearance --- Figure 1 is referenced in 3.1 but the image path appears inline before the caption text; Figure 2 similarly
- [x] Captions are self-explanatory --- Both figure captions are detailed and clear
- [ ] Text-heavy sections need visual support --- The graph construction (3.3) is entirely text-based and would benefit from a small diagram showing the top-k selection process

## What Works Well

1. **The theorem-proof format for the structural correspondence is excellent.** It elevates the LCA connection from an observation to a formal result. Despite the notation issue, the proof structure is correct and appropriate for a top venue.

2. **The six-phase experimental pipeline table (Section 4.1) provides outstanding at-a-glance clarity.** Mapping phases to hypotheses and tasks in a single table makes the empirical strategy transparent. This is exactly the kind of structural aid that helps reviewers follow complex methodology.

3. **The integration of prior data is handled with appropriate transparency.** Sections 4.5 and 4.6 explicitly note that precision/recall and absorption data come from prior experiments ("Recall and precision data come from our prior k-sparse probing experiments"). This honesty strengthens credibility and avoids the impression that all data is newly collected.

## Cross-Section Consistency Notes

- **Hypothesis numbering**: The hypotheses section (hypotheses.md: H1, H1b, H2, H3) does not align with the method section (H6-H10). The method section's Phase table references H6-H10 without defining them in the hypotheses section. This is a major inconsistency.

- **Intro consistency**: The intro's Key Results Preview (Section 1.5) promises H6-H10 results with specific predicted values ("precision@20 = X.XX"). The method section describes these as protocols without results. This is a critical mismatch.

- **Experiments consistency**: The experiments section (Section 5.2) mirrors the method section's protocol descriptions for H6-H10, also without results. Both sections describe experiments that were not executed.

- **Notation consistency**: The notation.md defines $G_{ij}$ as "inhibition from latent j to latent i" but the method section's inh_in equation uses $G_{j, \phi(f)}$ for incoming inhibition, which by this convention would be inhibition from phi(f) to j---the opposite direction. Verify and fix.

- **Related work consistency**: The related work section (2.3) correctly states "no prior work connects LCA to sparse autoencoders for language model interpretability." The method section's theorem formalizes this claim. This is consistent and well-supported.

- **Terminology**: "homeostatic rebalancing" is used consistently across method, glossary, and notation. "competitive suppression" is used consistently. "training-free" is hyphenated correctly throughout.
