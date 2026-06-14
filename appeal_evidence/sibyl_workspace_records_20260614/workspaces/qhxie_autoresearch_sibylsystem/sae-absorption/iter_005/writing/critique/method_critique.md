# Critique: Method (Section 3)

## Summary Assessment

The method section is comprehensive and well-structured, covering four distinct analysis phases with precise statistical specifications and pre-registered hypotheses. The epidemiological causal inference framework is rigorously described, and the threshold sweep designs are transparent. However, several technical descriptions are inconsistent with the experiments section, the absorption metric adaptation (Section 3.2.3) conflates two different detection approaches, and the section lacks any visual element despite the outline specifying a mediation path diagram (Figure 2) as belonging to this section's conceptual space.

## Score: 7/10
**Justification**: Strong statistical methodology descriptions, clear pre-registration, and good logical flow from Phase 1 through Phase 4. Loses points for: (1) inconsistencies between method and experiments descriptions, (2) the absorption metric adaptation uses terminology that differs from what the experiments section actually implements, (3) missing architecture/pipeline figure that a method section at this complexity level demands, and (4) several claims about the original proposal's H4 hypothesis silently dropped without acknowledgment. Reaching 8/10 requires resolving the method-experiments inconsistencies and adding a visual overview of the four-phase pipeline.

## Critical Issues

### Issue 1: Method describes "dominance ratio" but experiments implement "ablation dominance" -- terminology mismatch

- **Location**: Section 3.2.3, step 3 vs. experiments Section 4.2
- **Quote**: "A token is classified as absorbed if this dominant latent's activation magnitude exceeds the second-highest by a dominance ratio $\tau_{\text{dom}} \geq 1.0$."
- **Problem**: The method describes dominance as an activation magnitude ratio, but the experiments section (Table 5) reports absorption rates using the column header "$R_{\text{abs}}$ (dom $\geq 1.0$)" with rates that equal the FN rate exactly in every row. This means the dominance threshold of 1.0 is trivially satisfied (any nonzero dominant feature passes), so the "absorption rate" is simply the false-negative rate. The method section should make this operational equivalence explicit, or the threshold should be set higher to be a meaningful filter. Additionally, notation.md defines $\alpha_j(\mathbf{x})$ as an "ablation effect" computed via integrated gradients, but the method section describes "activation magnitude" -- these are different quantities.
- **Fix**: (a) Clarify whether the dominance comparison uses activation magnitudes or ablation effects (integrated gradients). (b) Note that at $\tau_{\text{dom}} = 1.0$, the metric reduces to the false-negative rate when any non-split feature is active. (c) Reconcile with notation.md's $\alpha_j(\mathbf{x})$ definition.

### Issue 2: Selectivity threshold inconsistency between method and proposal

- **Location**: Section 3.2.3, step 1
- **Quote**: "identify $k$ split latents -- SAE latents with selectivity $\geq 3.0$ (activation rate $\geq$ 5% on the target class and $\geq 3\times$ higher than on other classes)."
- **Problem**: The proposal (Section "Method, Phase 2, Step 2.2") specifies a threshold sweep for selectivity: "selectivity in {2, 3, 5, 10}". The outline also lists this sweep. But the method section hardcodes selectivity at 3.0 with no mention of a sweep. Meanwhile, the experiments section does not report any selectivity sweep results. Either the sweep was dropped (which should be acknowledged) or it was performed and results should be reported.
- **Fix**: Either (a) add the selectivity threshold sweep to the method description and ensure experiments reports the results, or (b) justify why a single threshold of 3.0 was chosen and acknowledge the deviation from the pre-registered plan.

## Major Issues

### Issue 3: Phase 2 model discrepancy between method and intro

- **Location**: Section 3.2.1
- **Quote**: "Gemma 2 2B was the intended target, but HuggingFace access restrictions at the time of experimentation necessitated the GPT-2 Small fallback."
- **Problem**: The intro section mentions "GPT-2 Small" directly as the model used, without hedging about it being a fallback. But the proposal's H2 hypothesis explicitly targets "Gemma 2 2B with Gemma Scope 16k SAEs at layers 8-17". The method section acknowledges this but frames it as a minor limitation, while the results show it fundamentally changed the experiment's outcome (0% cosine-calibrated absorption, 98% dead features). The method should more prominently flag that the model change invalidates the original H2 hypothesis as stated and that H2 is being tested on a different model-SAE pair than pre-registered.
- **Fix**: Add a paragraph explicitly stating that the H2 hypothesis is being tested on a different model than pre-registered, note how the change affects interpretation (smaller model, smaller SAE, different layer range 5/8/11 vs. 8/12/17), and cross-reference the limitations discussion (Section 5.4).

### Issue 4: Cosine-calibrated detection threshold mismatch

- **Location**: Section 3.2.3, step 4; Section 3.2.5
- **Quote**: "absorption is declared only when the dominant latent's decoder direction $\mathbf{d}_j$ has $\text{cos}(\mathbf{p}, \mathbf{d}_j) > \tau_{\text{cos}}$ with the probe direction $\mathbf{p}$"
- **Problem**: notation.md defines $\tau_{\text{cos}}$ with a default of 0.025. The threshold sweep in Section 3.2.5 lists $\tau_{\text{cos}} \in \{0.05, 0.10, 0.15, 0.20, 0.30\}$ -- the default value of 0.025 is not included in the sweep. The experiments section reports results at $\tau_{\text{cos}} = 0.1$ and $\tau_{\text{cos}} = 0.05$. These three sources disagree on which thresholds are canonical.
- **Fix**: Harmonize the threshold grid. Either update notation.md to remove the default, or include 0.025 in the sweep grid, or explain why the default was excluded.

### Issue 5: No figure or pipeline diagram in the method section

- **Location**: Entire Section 3
- **Quote**: "Figure 2 illustrates the mediation path model central to Phase 1." (line 3)
- **Problem**: Section 3 contains exactly one figure reference (Figure 2), and it is a forward reference to a diagram that appears in Section 4.1. For a method section describing four distinct analysis phases with complex dependencies, the absence of a pipeline/architecture overview figure is a significant gap. The outline's Figure & Table Plan does not include a method pipeline figure, but the complexity of the four-phase design warrants one. Reviewers at top venues expect a visual method overview when the pipeline has this many stages.
- **Fix**: Add a pipeline overview figure showing the four phases, their inputs (48 SAEs, 3552 cities, 420 SAEs, 26 letters), analysis methods, and outputs. This figure should appear near the top of Section 3.

### Issue 6: Sample size discrepancy -- 48 vs. 54 SAEs

- **Location**: Section 3.1.1 and throughout
- **Quote**: "The confound resolution analysis uses 48 Gemma Scope SAEs (Gemma 2 2B) from SAEBench" vs. the proposal's "iter_4 54-SAE dataset"
- **Problem**: The proposal repeatedly references 54 SAEs. The method and intro use 48 SAEs, with the explanation that "Six canonical SAEs lacking reported $L_0$ are excluded." This is methodologically sound, but the transition from 54 to 48 is not explicitly discussed as a deviation from the proposal. The outline section 3.1 says "48 Gemma Scope SAEs" which matches, but the intro says "48 Gemma Scope SAEs" while quoting the prior correlation as "r = -0.595 across 54 Gemma Scope SAEs" -- this juxtaposition is confusing.
- **Fix**: Add a brief note in Section 3.1.1 that the 54-SAE dataset from prior work reduces to 48 after excluding 6 SAEs with missing L0, and that the bivariate correlation on the 48-SAE subset is [value], not the previously reported r = -0.595 on 54 SAEs.

### Issue 7: H4 hypothesis silently dropped

- **Location**: Section 3.5 (hypotheses)
- **Quote**: Section lists H1, H2, H3 -- no H4.
- **Problem**: The proposal defines H4: "Early-type absorption (decoder-absent, no decoder direction within cosine > 0.3 of parent probe) dominates at > 50% of absorbed instances for knowledge-domain hierarchies." The experiments section jumps from H3 (Section 4.3) to H5 (Section 4.4, labeled as taxonomy correction). H4 is never mentioned. Either H4 was tested and dropped, or its numbering was absorbed into another hypothesis. The gap from H3 to H5 is jarring.
- **Fix**: Either (a) add H4 to the method section with a note about why it was not testable (e.g., 0% cosine-calibrated absorption made early/late type classification impossible), or (b) renumber H5 to H4 for clean sequencing.

## Minor Issues

- **Section 3.1.1**: "Six canonical SAEs lacking reported $L_0$ are excluded" -- what makes them "canonical"? This term is undefined. Replace with "six SAEs" or define what canonical means in this context.
- **Section 3.1.2**: "extends the prior analysis (which controlled for $\log m$ and $\ell$ but not $L_0$)" -- this parenthetical lacks a citation. Add "(Chanin et al., 2024)" or the specific prior analysis being referenced.
- **Section 3.1.4**: The Baron-Kenny four steps are numbered 1-4, but the numbering labels them as "Path a", "Path b", "Total effect c", "Indirect effect". The classical Baron-Kenny procedure has four *conditions*, not four paths. The numbering conflates regression equations with mediation conditions. Consider labeling as "Equation 1", "Equation 2", etc., with conditions described in text.
- **Section 3.1.5**: "High-absorption ($R_{\text{abs}} > 0.3$) and low-absorption ($R_{\text{abs}} < 0.1$)" -- the thresholds 0.3 and 0.1 are not justified. Why these cutoffs? The experiments section reports propensity matching with 6 pairs and Mahalanobis matching with 17 pairs, suggesting different cutoffs or a continuous matching approach was actually used.
- **Section 3.2.2**: The probe layers are 5, 8, 11 -- but the proposal specified layers 8, 12, 17 for Gemma 2 2B. The change to GPT-2 Small layers should be briefly justified (GPT-2 Small has 12 layers, so 5/8/11 spans early/mid/late).
- **Section 3.2.2**: "averaged over 3 random seeds (42, 123, 456)" -- the experiments section does not report standard deviations or seed-level results. If averaging is described in the method, the experiments should report variability.
- **Section 3.2.4**: "First-letter baseline: The standard Chanin et al. first-letter spelling absorption measurement on the same SAE provides a within-model reference point (literature range: 15--35% on Gemma Scope)." The literature range is for Gemma Scope, not GPT-2 Small. This comparison is potentially misleading. The experiments section reports first-letter rates of 85-93% on GPT-2 Small (discussion Section 5.2), which is outside the stated range.
- **Section 3.3.1**: "9 release families" -- the release families are never listed. A footnote or supplementary table would help reproducibility.
- **Section 3.3.1**: Architecture breakdown says "360 standard (L1), 54 JumpReLU, 6 unknown" = 420. The "unknown" category should be explained.
- **Section 3.4.1**: "the original implementation falls back to a global mean-when-active baseline" -- citation needed for which implementation (Chanin et al. code?).
- **Section 3.4.2**: Strategy C is described as "ground truth" -- this is misleading. The Chanin false-negative metric is an independent measurement, not ground truth. Rename to "Strategy C (independent validation)" or similar.
- **Section 3.6**: The final note ("Table 1... Figure 1... Figure 3...") is an orphaned forward-reference block that reads like editorial markup rather than polished prose. Integrate into the text or remove.
- **Section 3.6**: "pyGAM (generalized additive models)" -- the experiments section and scaling analysis may use a different GAM implementation. Verify consistency.
- **Terminology**: Section 3.2.3 uses "k-sparse probing" but the glossary defines "sparse probing" as a SAEBench metric. Clarify whether this is the same method or a different procedure.

## Visual Element Assessment

- [ ] Figures/tables match outline plan -- **FAIL**: The outline specifies Figure 2 (mediation path diagram) as appearing in Section 4.1, not Section 3. Section 3 references it but has no figures of its own. A pipeline overview figure is missing.
- [ ] All visuals referenced before appearance -- **PASS**: Figure 2 is referenced in the opening paragraph before it appears in Section 4.
- [ ] Captions are self-explanatory -- **N/A**: No figures in this section.
- [x] No text-heavy sections that need visual support -- **FAIL**: The four-phase pipeline (confound resolution, cross-domain, scaling surface, taxonomy) would benefit from a visual overview. The Baron-Kenny mediation model (Section 3.1.4) would benefit from a simple path diagram placed in the method section rather than only in results.

## What Works Well

1. **Pre-registered hypotheses with falsification criteria (Section 3.5)**: The explicit falsification thresholds for H1, H2, and H3 are specific, testable, and intellectually honest. The "$|r_{\text{partial}}| > 0.2$" go/no-go criterion and the "$3\times$ shuffled baseline" criterion for H2 demonstrate genuine commitment to letting data decide. This is rare in ML papers and will be appreciated by reviewers.

2. **Epidemiological method toolkit is well-selected and well-described**: The progression from partial correlations (Section 3.1.2) to width stratification (3.1.3) to Baron-Kenny mediation (3.1.4) to Rosenbaum sensitivity (3.1.5) to suppression diagnosis (3.1.6) is logically tight. Each method addresses a different threat to causal inference, and together they form a triangulation strategy that is more convincing than any single method alone. The VIF reporting ($\text{VIF}(\log L_0) = 1.09$) is a concrete diagnostic that strengthens credibility.

3. **Control design for cross-domain experiment (Section 3.2.4)**: The shuffled-hierarchy control is well-conceived -- it destroys hierarchical structure while preserving all other statistical properties of the data. Combined with the random-probe-direction control and the first-letter baseline, this three-control design provides strong diagnostic power for interpreting positive or negative results. The controls ultimately revealed the metric limitation, demonstrating their value.
