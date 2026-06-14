# Critique: Background and Related Work

## Summary Assessment
The related work section is well-organized across six subsections that effectively position the paper's contributions against existing literature. It covers SAE fundamentals, absorption definition, adjacent failure modes, architecture mitigation, RAVEL, and the evaluation gap. The final paragraph correctly identifies the gaps this paper fills. The section's primary weaknesses are (1) an inconsistency between the first-letter absorption rates reported here and those in the experiments section, (2) a missing transition from the outline's promise, and (3) several claims that lack specific evidence or citations.

## Score: 7/10
**Justification**: Solid coverage of the literature, well-structured subsections, and a clear gap statement. To reach 8/10, the section needs to fix the numerical inconsistencies with the experiments section, add missing citations for several claims, tighten the final paragraph to avoid previewing results that belong in the introduction, and address the terminology/notation issues flagged below. To reach 9/10, it would also need stronger logical threading between subsections and more explicit comparison of how each prior work's limitations create the specific gap this paper addresses.

## Critical Issues

### Issue 1: Inconsistent first-letter absorption rates between related work and experiments section
- **Location**: Section 2.2, paragraph 1
- **Quote**: "measure absorption rates of 15--35% across hundreds of Gemma Scope, Llama 3.2, and Qwen2 SAEs on the first-letter spelling task"
- **Problem**: This 15--35% range is cited as Chanin et al.'s finding across multiple models. The experiments section (Section 4.1) reports 27.1% at L24/16k and 17.7% at L24/65k for first-letter on Gemma 2 2B. Meanwhile, the introduction says "first-letter at 27.1%." These are broadly consistent with the 15--35% range. However, the proposal and earlier iterations reference "42.9% first-letter at L24" (see outline Section 4, "first-letter 42.9%") while the experiments section says 27.1%. The related work section should be aware that the 15--35% range from Chanin et al. is for GPT-2 Small and other models, not Gemma 2 2B. This distinction is never made explicit, creating ambiguity about whether the reader should expect the same range on the model studied in this paper.
- **Fix**: Add a parenthetical clarifying the model context: "measure absorption rates of 15--35% across hundreds of Gemma Scope, Llama 3.2, and Qwen2 SAEs on the first-letter spelling task (on GPT-2 Small and other models)." Also verify that the outline's "42.9%" figure is reconciled with the experiments section's "27.1%" -- these appear to be from different SAE configurations or measurement protocols, and the discrepancy should be resolved across the paper.

### Issue 2: Outline vs. experiments numerical mismatch propagated into cross-references
- **Location**: Outline Section 4 vs. Experiments Section 4.1
- **Quote**: Outline says "first-letter 42.9%" at L24; experiments section says "first-letter (27.1%, CI [24.5, 29.5])."
- **Problem**: The related work section's final paragraph states "We extend absorption measurement to four feature hierarchies" and cites the paper's contributions. If a reviewer checks the outline against the experiments, the 42.9% vs. 27.1% discrepancy will raise questions about data integrity. While this is technically a cross-section consistency issue, the related work's gap statement implicitly relies on these numbers being stable.
- **Fix**: Ensure the outline is updated to match the experiments section's final numbers. The related work section itself does not cite these specific numbers, but the editor should flag this for the outline.

## Major Issues

### Issue 3: Missing citation for Google DeepMind deprioritization claim
- **Location**: Section 2.6, paragraph 2
- **Quote**: "Google DeepMind's decision to deprioritize SAE research was driven partly by 10--40% performance degradation on safety-relevant downstream tasks (Smith et al., 2025)"
- **Problem**: The citation "Smith et al., 2025" appears here but is not referenced anywhere else in the paper. The introduction cites the same claim as "(Karvonen et al., 2025)." One of these citations is incorrect, or they refer to different sources for the same claim. A reviewer will flag this inconsistency.
- **Fix**: Verify which citation is correct. If both are valid (Karvonen et al. report the SAEBench findings and Smith et al. report the internal DeepMind decision), clarify the distinction. If the same claim has two different citations across sections, unify them. The introduction's "(Karvonen et al., 2025)" is more established since SAEBench is the well-known benchmark paper.

### Issue 4: Section 2.4 reads as a list, not an argument
- **Location**: Section 2.4, entire subsection
- **Quote**: "JumpReLU SAEs (Rajamanoharan et al., 2024a) use a learnable threshold activation... BatchTopK SAEs (Rajamanoharan et al., 2024b) relax the TopK constraint... Matryoshka SAEs (Bussmann et al., 2025) train nested dictionaries... OrtSAE (Korznikov et al., 2025) enforces orthogonality... Adaptive Temporal Masking (Li et al., 2025)..."
- **Problem**: This subsection catalogs six architectures in sequence without building a comparative argument. Each architecture gets a one-to-two sentence summary with an absorption number, but there is no organizing principle (e.g., ordered by mechanism type, by absorption reduction magnitude, or by the theoretical reason they reduce absorption). A reviewer at NeurIPS/ICML would see this as a laundry list rather than a synthesis. The final sentence about the first-letter limitation is the key point but arrives only after the reader has processed six disconnected descriptions.
- **Fix**: Reorganize around the *mechanism* of absorption reduction. Group architectures by strategy: (1) those that increase effective sparsity budget (BatchTopK, Matryoshka), (2) those that enforce decoder geometry constraints (OrtSAE, KronSAE), (3) those that modify training dynamics (Adaptive Temporal Masking, masked regularization). Lead with the organizing principle, then discuss each group. Move the "critical limitation" sentence (single-task evaluation) to the subsection's opening to frame why the catalog matters.

### Issue 5: "arXiv:2512.05534" is a bare arXiv ID without author attribution
- **Location**: Section 2.2, paragraph 2
- **Quote**: "a unified theory of sparse dictionary learning casts absorption as a natural consequence of the piecewise biconvex optimization landscape shared by all SAE variants (arXiv:2512.05534)"
- **Problem**: A bare arXiv identifier is not a proper citation. It lacks author names, which prevents the reader from evaluating the authority of the claim. Conference papers and preprints should be cited by author, not by arXiv number. This will be flagged in any formal review.
- **Fix**: Replace with a proper author citation, e.g., "Author et al. (2025)." If the reference is known, use the standard "Lastname et al., year" format.

### Issue 6: Bussmann et al. year inconsistency between related work and outline
- **Location**: Section 2.4 vs. Outline Section 2
- **Quote**: Related work: "Matryoshka SAEs (Bussmann et al., 2025)"; Outline: "Matryoshka (Bussmann et al., 2024)"
- **Problem**: The Matryoshka citation year differs between the related work section (2025) and the outline (2024). A reviewer checking references will flag this as sloppy.
- **Fix**: Verify the correct publication year and unify across all sections.

### Issue 7: Section 2.3 mentions several works without clear relevance to absorption
- **Location**: Section 2.3, multiple sentences
- **Quote**: "Feature non-canonicality (Leask et al., 2025) challenges the assumption that SAE latents are atomic units: meta-SAEs decompose latents into sub-features, and larger dictionaries discover qualitatively new latents missed by smaller ones."
- **Problem**: The connection between non-canonicality and absorption is not explicitly drawn. A reader is left wondering: does non-canonicality cause absorption? Does it interact with absorption measurement? Is it a confound? The same issue applies to "SAE dark matter" -- 50% of reconstruction error being linearly predictable is interesting but its relationship to absorption is unstated. A top-venue reviewer will ask "why is this here?"
- **Fix**: For each adjacent failure mode, add one sentence explicitly connecting it to absorption. For non-canonicality: "Non-canonicality is relevant because if latents are not atomic, the parent-child hierarchy assumed by absorption measurement may itself be an artifact of dictionary granularity." For dark matter: "Dark matter is relevant because the unexplained reconstruction error may mask absorption instances that fall below detection thresholds."

## Minor Issues
- **Section 2.1, paragraph 2**: "Rapid scaling followed:" is followed by three separate works in a single long sentence separated by semicolons. Break into shorter sentences for readability, or use a more structured presentation.
- **Section 2.1, paragraph 2**: "safety-relevant features such as deception and sycophancy" -- "sycophancy" should be "sycophancy" (verify spelling; the standard term in the literature is "sycophancy" but some papers use "sycophancy"). Confirm consistent spelling.
- **Section 2.2, paragraph 2**: "Theoretical analysis deepens this pessimism" -- "deepens this pessimism" is editorializing. Rephrase to a neutral framing: "Theoretical analysis reinforces this result" or "Subsequent theoretical work extends this finding."
- **Section 2.4**: "Narayanaswamy et al., 2026" -- a 2026 citation in a paper submitted in 2026 is valid only if the work is already publicly available (e.g., arXiv preprint). Verify this is not a future-dated citation error.
- **Section 2.4**: "KronSAE (2025)" -- missing author attribution. Should be "KronSAE (Author et al., 2025)."
- **Section 2.4**: "SynthSAEBench (2026)" in Section 2.6 -- also missing author attribution.
- **Section 2.5**: "Resolved Attribute Value Estimation for Language models" -- the "m" in "models" should be capitalized if it is part of the formal acronym expansion. Check the original paper.
- **Section 2.6**: The final paragraph ("This paper addresses these gaps") previews paper contributions. The outline's transition note says this section should transition to methodology, not recapitulate contributions. The introduction already previews contributions comprehensively. Consider shortening this to a single transitional sentence pointing forward to Section 3.
- **Section 2.2**: The glossary specifies "feature absorption" on first use, then "absorption" thereafter. The section correctly uses "Feature absorption" as the subsection title but immediately uses "feature absorption" in the body. Consistent with glossary -- no issue.
- **Notation check**: $\mathbf{x} \in \mathbb{R}^d$, $\mathbf{z} \in \mathbb{R}^m$, $\hat{\mathbf{x}} = \mathbf{W}_{\text{dec}} \mathbf{z} + \mathbf{b}_{\text{dec}}$, $z_i$, $\mathbf{d}_c$, $\mathbf{w}_p$, $L_0$ -- all consistent with notation.md.
- **Notation check**: $\tau_{\cos}$ and $\tau_{\text{gap}}$ in Section 2.2 are not defined in notation.md. Add them or reference them as "from Chanin et al.'s protocol."
- **Glossary check**: "JumpReLU", "BatchTopK", "Matryoshka", "Gemma Scope", "SAEBench", "RAVEL" -- all match glossary capitalization and formatting.

## Visual Element Assessment
- [x] No figures/tables planned for this section (confirmed in outline: "None")
- [x] N/A -- no visuals referenced
- [x] N/A -- no captions needed
- [x] The section is text-heavy but appropriately so for a related work section; no visual support needed

## What Works Well
- **Section 2.2 (Feature Absorption)** provides a genuinely useful technical explanation of the absorption mechanism, including the $L_0$ cost argument and the decoder similarity explanation. The sentence "For a parent-child pair $(p, c)$, an SAE with absorption encodes both concepts at $L_0$ cost $+1$ (child only), while absorption-free encoding requires $+2$" is clear, precise, and well-motivated -- it gives the reader the key intuition without hand-waving.
- **Section 2.5 (RAVEL)** effectively motivates why RAVEL's city hierarchies are natural extensions of the first-letter task by enumerating the specific structural differences (class count, balance, co-occurrence pattern, semantic richness). The final sentence -- "RAVEL hierarchies are imbalanced, multi-level, and grounded in real-world knowledge" -- directly sets up the paper's contribution.
- **Section 2.6 (Evaluation Gap)** clearly identifies the three missing dimensions (cross-domain, cross-layer, safety contribution) that this paper addresses. The gap identification is specific and falsifiable, which is exactly what a related work section should achieve.
