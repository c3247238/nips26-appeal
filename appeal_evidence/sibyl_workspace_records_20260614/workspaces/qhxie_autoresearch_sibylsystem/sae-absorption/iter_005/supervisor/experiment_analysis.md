# Experiment Result Analysis

## Key Results Summary

### H1: Absorption-Quality Causal Chain (Confound Resolution)
- Sparse probing partial r **strengthened** from -0.664 to -0.746 (p = 1.16e-09) after controlling for L0 -- a suppression effect, the opposite of what was predicted
- SCR partial r = -0.570 (p = 6.57e-05), TPP partial r = -0.331 (p = 0.022) after L0 control
- Rosenbaum sensitivity: Gamma = 2.65 on Mahalanobis matching (17 pairs), but **all within-width matching strategies fail** (p > 0.10 for every metric)
- Mediation: Full Baron & Kenny for SCR (Sobel p = 2.94e-04) and TPP (Sobel p = 3.73e-02); sparse probing proportion mediated = 4.785 (incoherent, total effect near zero)
- Bradford Hill assessment: 3 strong, 5 moderate, 0 weak criteria
- Sample size: n = 48 SAEs (Gemma 2 2B), 6 canonical SAEs excluded due to missing L0

### H2: Cross-Domain Absorption (GPT-2 Small)
- Dominance-based absorption: 51-85% on knowledge hierarchies, but **100% on shuffled controls** -- metric invalidated
- Cosine-calibrated absorption: 0% across all thresholds and domains
- Root cause: GPT-2 Small SAE has 98% dead features; a single "super-absorber" feature (8213) dominates all token positions regardless of label structure
- Model choice was a fallback from Gemma 2B due to gated access

### H3: Absorption Scaling Surface
- GAM interaction: p = 3.11e-15 on N = 420 SAEs from SAEBench
- R-squared progression: 0.488 (linear) -> 0.620 (additive GAM) -> 0.693 (interaction GAM)
- Phase boundary at log2(L0) in [2.7, 3.8]: absorption increases dramatically at 1M width with low L0 but vanishes at high L0
- 420 SAEs includes 360 standard, 54 JumpReLU, 6 unknown architecture

### H5: Taxonomy Correction
- **CRITICAL REPORTING ERROR**: final_results_summary.md reports "corrected rate = 92.3%, delta = 0.0%" but P4_taxonomy_correction.json shows corrected rate = **19.2%** (delta = -73.1 pp)
- Corrected Type II: 15.4% (down from 88.5%), with 19/26 letters changing classification
- Chanin false-negative metric: 73.1% of letters show any-absorption (independent validation)
- The original 92.3% was inflated by the n_comparison_tokens=0 fallback to a biased global baseline

## Debate Perspectives Summary

- **Optimist**: The suppression effect (sparse probing strengthening after L0 control) is "the single most compelling result" and exceeds the most optimistic prediction. The 420-SAE scaling surface is "extraordinary." The cross-domain metric failure is itself a publishable boundary-condition finding. Two papers' worth of material. Bottom line: strong publishable story.

- **Skeptic**: The within-width matching null "directly contradicts the headline claim" -- cross-width Rosenbaum Gamma = 2.65 is inflated because matched pairs span width strata. Proportion mediated for sparse probing (4.785) is mathematically incoherent. n = 48 with 50+ tests and no multiple comparison correction is underpowered. The taxonomy reporting error (92.3% vs actual 19.2%) is a fatal data integrity issue. Cross-domain metric is invalidated, not partially supported. Verdict: one strong finding (scaling surface), one ambiguous (confound resolution), one failed (cross-domain), one self-contradicting (taxonomy).

- **Strategist**: PROCEED. Two publication-ready contributions (confound resolution + scaling surface) require zero additional GPU compute. Writing is the critical path, not more experiments. Phase 2 negative result is a publishable methodological insight. Do NOT pursue Gemma 2B replication (deferred). Optional: IG-based absorption on GPT-2 (2-3 GPU-hours) in parallel with writing. Estimated paper completion: 4-6 hours of writing.

- **Comparativist**: Contribution margin: confound resolution = STRONG (first application of epidemiological causal methods to SAE evaluation, no prior art), scaling surface = STRONG (first formal 2D surface, N = 420), cross-domain = MARGINAL (informative negative on wrong model), taxonomy = MODERATE. No concurrent paper addresses the same three questions. Venue: workshop-ready now (70% acceptance), conditional on Gemma 2B for main conference (35% NeurIPS/ICML).

- **Methodologist**: Phase 1 and Phase 3 methodology is GOOD. Phase 2 is POOR due to model fallback (GPT-2 Small vs proposed Gemma 2B). The taxonomy reporting discrepancy is a credibility-threatening data integrity issue. Missing ablations: architecture-specific GAM (zero-GPU, 15 min), excluded SAE sensitivity check, layer-specific mediation. Reproducibility score: 3.6/5.

- **Revisionist**: The sparse probing suppression effect is a qualitative reversal of the expected confounding direction (major surprise). Cross-domain metric failure is devastating (major surprise). Taxonomy correction is massive -- 73.1 pp drop, not "minimal" (moderate surprise). Within-width matching null constrains causal interpretation. Three originally conflated questions must be separated: association (yes), causation independent of width (uncertain), cross-domain generalization (unknown). Revised framing: "Feature absorption in SAEs: how robust is the causal chain, how inflated is the prevalence, and what are the boundary conditions for cross-domain measurement?"

## Analysis

### 1. Method Feasibility

The core methods work well for two of three contributions. Phase 1 (confound resolution) successfully applied epidemiological causal inference methods -- partial correlations, mediation analysis, Rosenbaum bounds, Bradford Hill criteria -- producing novel and informative results. Phase 3 (scaling surface) leveraged existing SAEBench data for a clean, high-powered analysis (N = 420). Phase 2 (cross-domain absorption) failed due to the model fallback: GPT-2 Small with 98% dead SAE features cannot support meaningful absorption measurement on knowledge hierarchies. The dominance-based metric is fundamentally invalidated by its inability to distinguish real from shuffled labels. Phase 4 (taxonomy correction) succeeded methodologically but has a critical data integration bug misreporting the results.

The experimental infrastructure works. The zero-GPU design for Phases 1 and 3 was validated as efficient and effective. The probe training for Phase 2 completed successfully. The failure point was not methodology but the model choice forced by access constraints.

### 2. Performance vs. Baselines

**Phase 1**: The confound resolution exceeds expectations. The original Chanin et al. correlation of r = -0.595 (uncontrolled) was expected to weaken to |0.3-0.45| after L0 control. Instead, sparse probing strengthened to r = -0.746 -- a suppression effect that is the most informative finding of the iteration. SCR (-0.570) and TPP (-0.331) also retain significance. Only unlearning drops to non-significance. This represents a genuine advance over the prior state of the art.

**Phase 3**: The 420-SAE scaling surface with p = 3.11e-15 interaction far exceeds any prior analysis. Chanin et al. showed visual scatter plots (Figures 9b/9c) without formal testing. The R-squared improvement from 0.488 to 0.693 demonstrates substantial variance captured by the interaction model.

**Phase 2**: The cross-domain results are strictly worse than baseline. The shuffled control rate of 100% means the dominance-based metric has zero discriminative power. The cosine-calibrated rate of 0% may be too strict. No valid measurement was achieved.

### 3. Improvement Headroom

Within the current direction, there is clear improvement headroom through targeted, low-cost actions:

- **Fix data integrity issues** (< 1 hour): The taxonomy reporting error, phase_boundary_detected inconsistency, and n_full_mediations count are all fixable immediately.
- **Architecture-specific GAM** (15 minutes, zero GPU): Testing the interaction within standard SAEs alone (n = 360) would rule out the architecture confound.
- **Power analysis for within-width matching** (30 minutes): Quantifying whether the within-width null is due to low power vs. genuine absence would sharpen the causal interpretation.
- **FDR correction** (30 minutes): Applying multiple comparison correction would identify which Phase 1 findings survive stringent adjustment.

For the cross-domain contribution, the headroom requires either Gemma 2B model access (uncertain) or implementing the IG-based absorption method on GPT-2 (2-3 GPU-hours). Both are feasible but not guaranteed to produce positive results.

### 4. Time-Cost Tradeoff

Continuing the current direction is highly efficient because two of three contributions are publication-ready with zero additional GPU compute. The critical path is now writing, not experimentation. The estimated cost to reach a submittable paper:

- Writing: 4-6 hours
- Data integrity fixes: 1 hour
- Architecture ablation + power analysis: 1 hour
- Optional IG experiment: 2-3 GPU-hours (parallel with writing)

Pivoting to an alternative would require abandoning two strong contributions (confound resolution and scaling surface) that have no equivalent in the pivot candidates. Alternative 1 (Immunodominance) addresses different questions. Alternative 2 (Phase Diagram) is already subsumed by the H3 result. Alternative 3 (Honest Audit) is only appropriate if H1 and H2 were both falsified -- H1 survived. Alternative 4 (Dictionary Coverage) would require new multi-model experiments.

### 5. Critical Objections Assessment

The skeptic raises four serious concerns. Here is my assessment of each:

**Within-width matching null**: This is the strongest objection. The inability to detect absorption-quality differences within a given width stratum means the causal claim cannot be fully separated from width effects. However, this does not invalidate the association finding. The resolution: frame the paper around "robust association after confound control" rather than "causal chain." The suppression effect is informative regardless of the within-width null.

**Proportion mediated = 4.785**: Valid objection. Sparse probing mediation is uninterpretable and must be excluded from the mediation narrative. The paper should rest the mediation claim on SCR and TPP only. This is addressable in writing.

**Taxonomy reporting error**: Valid and critical. This is a bug in the integration script, not a judgment call. It must be fixed before any writing. The actual correction (92.3% -> 19.2%) is far more dramatic than reported and should be promoted rather than suppressed.

**Architecture confound in scaling surface**: Valid concern but likely survivable. With 360 standard SAEs and the extremely small p-value, the interaction should hold within-architecture. A 15-minute ablation resolves this.

None of the skeptic's objections are fatal to the overall research direction. They constrain the strength of certain claims and require honest reporting, but the core findings (suppression effect, scaling surface interaction, taxonomy correction) are robust.

## Decision Rationale

The evidence overwhelmingly supports PROCEED for the following reasons:

1. **Two independently strong contributions are ready**: The confound resolution (H1) with its unexpected suppression effect and the scaling surface (H3) with p = 3.11e-15 on N = 420 SAEs are both novel findings with no prior art in the SAE literature. Together they form a coherent narrative about characterizing absorption rigorously.

2. **The failed Phase 2 does not block publication**: The cross-domain metric failure is a legitimate negative/diagnostic finding. When reframed as "the standard absorption metric does not transfer to small models with high SAE sparsity" rather than "cross-domain absorption does not exist," it becomes a methodological contribution about measurement validity boundaries.

3. **No alternative direction is stronger**: All four backup candidates are weaker than the current combination. Alternative 2 (Phase Diagram) is already subsumed by H3. Alternative 3 (Honest Audit) only applies if H1 were falsified, which it was not. Alternatives 1 and 4 address different, less pressing questions.

4. **The critical path is writing, not experimentation**: Zero additional GPU hours are required for the core paper. All remaining improvements (data fixes, architecture ablation, power analysis) are low-effort zero-GPU tasks that can be completed during writing.

5. **Competitive timing matters**: The Comparativist confirms no concurrent work addresses the same questions. The Masked Regularization paper (April 2026) is complementary. Delaying for Gemma 2B access or additional experiments risks losing the novelty window.

6. **All six debate perspectives converge on PROCEED**: The synthesis scores the results 6.5/10 overall (7.5/10 without the failed Phase 2). Every perspective that rendered a verdict -- Optimist, Strategist, Comparativist, Revisionist, and the Synthesis -- recommends proceeding. Even the Skeptic, while raising important caveats, does not argue for pivot but rather for honest presentation of limitations.

**Required pre-writing fixes**: (a) taxonomy reporting error in final_results_summary.md and final_results.json, (b) phase_boundary_detected inconsistency, (c) n_full_mediations count. These are data integrity issues, not scientific disagreements.

**Required narrative adjustments**: (a) Weaken "causal" to "robust association" for H1, (b) present within-width null transparently, (c) reframe Phase 2 as diagnostic/methodological rather than positive, (d) report both Rosenbaum Gamma values (cross-width 2.65, within-width 1.0), (e) exclude sparse probing from mediation claims.

## DECISION: PROCEED
