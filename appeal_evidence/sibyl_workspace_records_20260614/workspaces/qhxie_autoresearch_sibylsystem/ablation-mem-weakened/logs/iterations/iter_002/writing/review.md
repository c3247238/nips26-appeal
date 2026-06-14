# Writing Quality Review

## Summary

This paper proposes that the decoder correlation matrix G = W_dec^T W_dec is exactly the inhibition matrix from Rozell et al.'s Locally Competitive Algorithm (LCA), providing a mechanistic explanation for feature absorption in sparse autoencoders as competitive suppression. The authors construct a local inhibition graph from decoder correlations, describe its properties, and propose homeostatic rebalancing as a training-free repair. The paper integrates prior empirical findings (H1--H5) from GPT-2 Small into the competitive suppression framework and proposes five validation experiments (H6--H10) to test the theory's predictions.

## Detailed Assessment

### Structural Coherence: 5/10

The paper has a fundamental structural problem: it is a **proposal paper masquerading as a results paper**. Sections 5.2--5.6 describe experiments H6--H10 as validation protocols with predicted outcomes, but these experiments have not been executed. The abstract claims "we show that the decoder correlation matrix... is exactly the inhibition matrix" and "we construct a local inhibition graph," but the actual empirical content consists entirely of prior null results (H1--H5) that the new framework reinterprets. This creates a jarring mismatch between the paper's framing and its content.

The transition from Section 5.1 (actual prior results) to Section 5.2 (proposed experiments with predicted outcomes) is particularly problematic. The reader encounters detailed methodology for H6--H10 with predictions like "Precision@20 >= 0.10" and "parent firing increases by >20%," but these are speculative. The paper would be more coherent if framed as a theoretical contribution with empirical context, rather than a full empirical paper.

The abstract-to-content mismatch is severe: the abstract reads like a completed study, but the body reveals that the core validation experiments are future work. This is not a minor issue --- it is the central structural flaw of the manuscript.

### Notation & Terminology Consistency: 8/10

Notation is generally consistent with `notation.md`. Key symbols (G, W_dec, W_enc, z, a, d_dict, d_model) are used correctly throughout. The LCA dynamics equation appears three times (Sections 2.3, 3.1, 3.2) with identical formatting, which is good for readability.

**Specific violations:**
- Section 4.5 uses `inh_in(f)` but `notation.md` defines it as `inh_in(i)` (indexed by latent, not feature). The paper switches between feature index f and latent index phi(f) without always clarifying.
- Section 5.1, Table 3: "HIGH (>=10%)" uses >= but `notation.md` defines F_HIGH as A(f) > 0.10 (strict inequality). The table uses >=0.10 which is inconsistent with the notation definition.
- The paper uses "delta-corrected" and "baseline-corrected" interchangeably in places (Section 5.1 vs. Section 6.2). `glossary.md` prefers "delta-corrected."
- Section 1.5 says "21/26 features at 1.0 at layer 4" but `precision_recall_analysis.json` shows n_precision_one = 22 at k=1 and 21 at k=5. The paper does not specify which k_probe this refers to.

### Claim-Evidence Integrity: 5/10

This is the weakest dimension. The paper makes many claims that are not supported by the data it presents.

**Unsupported or misleading claims:**

1. **"Precision is invariant" (Section 5.1, Table 5):** The paper states "21/26 features at layer 4 and 25/26 at layer 8 achieve perfect precision (1.0)" at k_probe = 5. However, `precision_recall_analysis.json` shows at k=5, layer 4 has n_precision_one = 21 (correct), but layer 8 has n_precision_one = 25 (correct). The issue is that at k=1, layer 4 has only 22/26 with precision=1.0, and the paper does not specify k. More importantly, the claim "precision is invariant" is overstated --- precision varies from 0.818 to 1.0 at k=5 for layer 4 (std = 0.054). "Nearly invariant" would be more accurate.

2. **"Feature U (24.2% absorption) still steers 100%" (Section 5.3, Table 2):** `correlation_report_full.json` shows U's steering success at s=50 is 1.0 at both layers 4 and 8. This is correct.

3. **"H1b significant at layer 8" (Section 5.1, Table 1):** The paper correctly notes that H1b does NOT survive multiple comparison correction (Bonferroni p = 0.334; BH-FDR q = 0.167). However, the abstract says "delta-corrected steering revealed a negative trend at layer 8" without mentioning the correction failure. This is borderline --- "trend" is accurate, but readers may miss the correction issue.

4. **"The LCA--SAE structural correspondence is exact" (Section 6.1):** This is only true for tied-weight SAEs. The paper acknowledges this in Section 3.1 ("Even with untied weights... the structural correspondence holds approximately") but the strong claim in Section 6.1 omits this caveat.

5. **H6--H10 predictions are presented as if they were results:** Sections 5.2--5.6 describe experiments with predicted outcomes ("Precision@20 >= 0.10", "r < 0, p < 0.05") but no actual data. The paper should clearly mark these as "proposed experiments" or "predictions" rather than presenting them in the Results section.

6. **"H6--H10 not executed" is buried in Limitations (Section 7.1, item 5):** This should be front-and-center in the abstract and introduction, not hidden in limitations.

### Visual Communication: 6/10

The paper references 8 figures and 5 tables, but many are figure descriptions (`.md` files) rather than actual rendered figures. The figures that exist (`fig2_absorption_rates.pdf` through `fig7_precision_recall.pdf`) are data-driven visualizations from prior experiments, not from the new framework.

**Issues:**
- Figure 1 and Figure 2 are described as `.md` files (`fig1_lca_correspondence_desc.md`, `fig6_suppression_mechanism_desc.md`) --- these are text descriptions, not actual figures. The paper references them as if they were rendered diagrams.
- The outline calls for Figures 3--5 (H6--H10 results) but these do not exist because the experiments were not run.
- Table 1 (hypothesis tests) is clear and well-formatted with proper multiple comparison corrections.
- Table 2 (integration of prior findings) is effective but some "supporting evidence" entries are circular (e.g., "H1b significant at L8" is used to explain "Layer 8 effect stronger than layer 4," but H1b itself does not survive correction).
- The paper would benefit from a conceptual diagram showing the inhibition graph construction pipeline, but the referenced Figure 1 is only a text description.

**Missing visual support:**
- No actual rendered Figure 1 (LCA-SAE correspondence diagram)
- No actual rendered Figure 2 (competitive suppression mechanism)
- No figures for H6--H10 (because experiments not executed)

### Writing Quality: 7/10

The writing is generally clear and professional. Sentences are well-constructed, jargon is used appropriately, and the paper avoids most banned patterns.

**Banned patterns found:**
- "Moreover" appears in Section 6.4 ("Moreover, our approach complements these solutions"). This is a mild violation but should be removed per the banned patterns list.
- "Furthermore" does not appear.
- "In recent years" does not appear.
- "It is worth noting that" does not appear.

**Unclear or problematic sentences:**

1. Section 1.1: "Some research groups have reportedly deprioritized SAE research after finding negative results on downstream tasks." --- "Reportedly" is vague. Cite a source or remove.

2. Section 1.2: "A key insight, which to our knowledge has not been articulated in the SAE literature, is that for SAEs with tied encoder--decoder weights..." --- The hedging "to our knowledge" is appropriate, but the paper later states more strongly that "no prior work connects LCA to LLM SAEs" (Section 2.3). These should be consistent.

3. Section 3.1, Proof: "The SAE forward pass z = ReLU(W_enc a + b_pre) approximates the LCA steady-state where du/dt = 0" --- This is hand-wavy. The approximation quality depends on the time constant tau and the dynamics convergence rate. A more careful statement would acknowledge this.

4. Section 5.1: "The distribution is strongly right-skewed: 18--26 of 26 features per layer show absorption rates below 10%." --- Actually, the data shows 18 (L4), 22 (L8), 21 (L10), 26 (L0). The range "18--26" is correct but the phrasing "18--26 of 26" is awkward. Better: "Between 18 and 26 features per layer show absorption below 10%."

5. Section 8.2, Contribution 2: "It predicts known absorption pairs with enrichment over chance and identifies at-risk features before running absorption metrics." --- This claims the graph "predicts" and "identifies" but these capabilities have not been empirically validated. The paper has not executed H6 or H8.

**Passive voice:** Generally well-controlled. The paper uses active voice effectively ("We show that...", "We construct...").

## Issues for the Editor

1. **[Critical] Paper claims to present validated results but core experiments are unexecuted.** The abstract, introduction, and conclusion all frame the paper as presenting a validated framework, but H6--H10 are described as "proposed validation experiments" with predicted outcomes. The paper needs a fundamental reframing: either (a) frame it as a theoretical contribution with empirical context and proposed experiments, or (b) execute H6--H10 before submission. **Fix:** Restructure the paper as a "theoretical framework paper" with a clear "Proposed Validation" section, not a Results section with predicted outcomes. Move H6--H10 descriptions to a "Future Work" or "Proposed Experiments" section. Rewrite the abstract to accurately reflect that the framework is theoretical with proposed (not validated) predictions.

2. **[Major] Abstract misrepresents the paper's empirical status.** The abstract says "we show that...", "we construct...", "we also propose..." but the empirical validation of the core claims (graph predicts absorption, inhibition explains asymmetry, homeostatic rebalancing works) is absent. **Fix:** Rewrite the abstract to clearly state: (1) we establish the theoretical LCA-SAE correspondence, (2) we show how it explains prior empirical findings, (3) we propose validation experiments to test specific predictions. Do not imply the predictions have been validated.

3. **[Major] H1b significance claim is borderline misleading.** The paper correctly notes in Table 1 and text that H1b does not survive multiple comparison correction, but the abstract says "delta-corrected steering revealed a negative trend at layer 8" without the correction caveat. Readers skimming the abstract may conclude the effect is robust. **Fix:** Add "(uncorrected p = 0.028; does not survive Bonferroni or BH-FDR correction)" to the abstract's mention of H1b, or remove the H1b mention from the abstract entirely.

4. **[Major] "Exact" structural correspondence is overstated for standard SAEs.** The paper repeatedly calls the correspondence "exact" but this only holds for tied-weight SAEs, which are not the standard case. Section 3.1 acknowledges the approximation for untied weights, but Sections 6.1 and 8.1 revert to "exact" without the caveat. **Fix:** Add "for tied-weight SAEs" or "(exact for tied weights, approximate for standard untied SAEs)" whenever claiming exactness.

5. **[Minor] "Precision is invariant" is overstated.** Precision varies from 0.818 to 1.0 at k=5 (std = 0.054). "Nearly invariant" or "precision shows much less variance than recall" would be more accurate. **Fix:** Change "precision is invariant" to "precision is nearly invariant" throughout, and add the range (0.818--1.0) to support the claim.

## What Works Well

1. **Table 2 (Integration with Prior Findings)** is an excellent piece of scientific communication. It maps each empirical observation to a mechanistic explanation in a clear, scannable format. This is the paper's strongest section for demonstrating the value of the theoretical framework.

2. **The LCA proof in Section 3.1** is concise, well-structured, and mathematically clear. The correspondence between SAE forward pass and LCA steady-state is articulated cleanly, and the implication paragraph effectively bridges theory to mechanism.

3. **Section 6.4 (Relationship to Existing Solutions)** effectively positions the work relative to Matryoshka SAEs, OrtSAE, and HSAE. The explanation of why each architectural solution works through the lens of inhibition (reducing G_ij) is intellectually satisfying and demonstrates the explanatory power of the framework.

SCORE: 6
