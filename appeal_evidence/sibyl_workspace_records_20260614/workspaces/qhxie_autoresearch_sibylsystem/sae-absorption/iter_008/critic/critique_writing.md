# Critique: Writing

**Reviewer:** sibyl-critic
**Date:** 2026-04-15
**Target:** `current/writing/paper.md`, `current/writing/review.md`

---

## Overall Assessment: 7/10

The paper is well-structured, evidence-forward, and refreshingly honest about its limitations and negative results. The writing quality exceeds the mechanistic interpretability norm. However, several problems prevent publication readiness: the primary claim is framed too strongly given the probe quality confound, missing figures leave two key contributions without visual support, broken cross-references signal incomplete preparation, and the abstract overclaims relative to the evidence.

---

## Strengths

### S1. Evidence-forward paragraph structure
Nearly every paragraph opens with a concrete finding or specific number. The Introduction leads with "Absorption rates on the same sparse autoencoder vary 15-fold depending on which model layer is measured" -- not with generic motivation. This is exactly the style that makes a paper efficient to review.

### S2. Transparent limitation discussion
Section 4.3 (Probe Quality Confound) names three specific confound mechanisms with quantitative backing. Section 8.4 orders limitations by severity. This is unusually honest for the field and will earn reviewer goodwill.

### S3. Clean section transitions
Each section ends with a bridge to the next. "Having established that absorption rates vary by layer and hierarchy, we now provide causal evidence..." These transitions create narrative momentum and make the 9-section structure feel coherent rather than fragmented.

### S4. Negative results get first-class treatment
Section 7 dedicates equal rigor to GAS, CMI, and Tax failures as Sections 4-5 give to positive results. Each negative result includes the metric, sample size, and interpretation of why the approach failed.

---

## Weaknesses

### W1. CRITICAL -- The abstract and introduction overclaim relative to the evidence

The abstract states: "measured absorption rates differ significantly across hierarchy types (Kruskal-Wallis p=0.005, 4 of 6 pairwise comparisons significant at p<0.05)."

But the abstract also states: "this comparison is confounded by differential probe quality (rho=-0.756 between probe F1 and false negative rate)."

These two sentences are in tension but the abstract presents them sequentially, giving the impression that the confound is a minor caveat rather than an alternative explanation for the primary result. A reviewer reading the abstract will note that the paper simultaneously claims significance AND acknowledges a confound that could explain the significance. The framing should either:
1. Lead with the unconfounded finding (layer-dependence) as the primary result, or
2. Present cross-domain variation explicitly as "suggestive evidence of hierarchy-dependent absorption, confounded by differential probe quality"

The Introduction's "Four contributions" list places "Cross-domain absorption measurement" as the second contribution. Given the confound, it should be presented with explicit qualification: "suggestive cross-domain variation, pending resolution of the probe quality confound."

### W2. CRITICAL -- Two sections lack visual support

Section 5.1 (Activation Patching) presents the paper's strongest causal evidence (d=1.33) with only a truncated table. No paired dot plot, no violin plot, no visual showing the per-word treatment-vs-control separation. For a causal result, visual communication is essential: the reader should see at a glance that child-zeroing consistently produces more recovery than control across most words.

Section 5.2 (Hedging Decomposition) presents the "near-tautological hedging" finding as pure text. A stacked bar chart (7.9% strict / 86.2% compensatory / 5.9% persistent) would communicate this insight instantly.

Both figures can be generated from existing JSON data (`phase0/activation_patching_full.json`, hedging data) at zero GPU cost.

### W3. MAJOR -- Broken cross-references signal incomplete preparation

1. Section 8.2 cites "Table 7 in Section 4.4 of the extended results" -- neither Table 7 nor Section 4.4 exists in the manuscript.
2. Section 3.5 references "Section 8.6" for future patching at L24 -- Future Directions is Section 8.5; no 8.6 exists.
3. Table 5 says "full results in Appendix" -- no appendix text exists.

These are not cosmetic issues. A reviewer encountering "Table 7 in Section 4.4" will search the paper, fail to find it, and flag the manuscript as carelessly prepared. At a top venue, this erodes trust in the data reporting.

### W4. MAJOR -- The title no longer matches the content

The title is "The Absorption Tax: Layer-Dependent and Hierarchy-Dependent Feature Absorption Across Semantic Domains in Sparse Autoencoders."

The Absorption Tax is a theoretical framework that comprehensively FAILED (rho=-0.20, concordance 50%). It is relegated to Appendix D. Leading with "The Absorption Tax" in the title misleads the reader about the paper's actual contribution, which is the layer-dependence and cross-domain measurement.

A more accurate title would be: "Feature Absorption Varies 15-Fold Across Model Layers and Semantic Domains: A Cross-Domain Characterization of SAE Failure Modes." Or simply: "Beyond First-Letter Spelling: Cross-Layer and Cross-Domain Characterization of SAE Feature Absorption."

### W5. MAJOR -- The claim "hierarchy type explains more variance than architecture choice" is unsupported

This claim appears in the Discussion (Section 8.1) and Conclusion (Section 9): "hierarchy type dominates (p=0.005) -- suggesting that the input's semantic structure governs absorption more than the SAE's training objective."

This comparison is invalid because:
- The hierarchy p=0.005 comes from L24 cross-domain data with 8 Gemma Scope SAEs
- The architecture p=0.87 comes from L12 with 4 different SAE architectures at mismatched widths and L0

These are different tests on different data at different layers with different confounds. Comparing p-values from non-nested tests to make causal claims about which factor "matters more" is a statistical fallacy.

### W6. MAJOR -- Abstract length exceeds NeurIPS word limit

The abstract is approximately 250 words. NeurIPS requires abstracts under 250 words (or sometimes 200 for workshop papers). More importantly, the abstract tries to cover all four contributions plus all three negative results plus the probe quality confound. This density reduces readability.

The abstract should lead with the two strongest, unconfounded findings:
1. Layer-dependent absorption (15x variation, unconfounded)
2. Interventional causal evidence (d=1.33, strong effect)

Cross-domain variation (confounded), hedging decomposition (one hierarchy only), and the three negative detectors can be covered in one sentence each or deferred to the Introduction.

### W7. MINOR -- Section 8.2 is overly speculative

The "Layer-Position Mechanism Hypothesis" introduces representation-sharpening and two falsifiable predictions without any supporting evidence beyond the observation that absorption spikes at L24 (near the final layer). The paragraph "At earlier layers, the residual stream carries distributed representations that SAEs can encode without strong hierarchical competition; at layer 24, the representation sharpens toward specific token predictions" is a plausible narrative but has zero empirical support in this paper.

The paper correctly labels this as a hypothesis, which is appropriate. But the two falsifiable predictions (penultimate-layer spike in deeper models, entropy/norm sharpening) are generic enough to be unfalsifiable in practice -- any number of alternative mechanisms could produce or fail to produce these patterns.

### W8. MINOR -- Glossary inconsistency with published terminology

The paper uses "feature absorption" throughout, which is the Chanin et al. (2024) term. But SAEBench (Karvonen et al., 2025) uses "absorption" as a metric name (lower = better). The paper's absorption rate (higher = worse) and SAEBench's absorption score (lower = better) are related but not identical. A brief clarification of the relationship between this paper's AR metric and the SAEBench absorption score would prevent confusion.

---

## Writing Quality Summary

| Dimension | Score | Comment |
|-----------|-------|---------|
| Structural coherence | 8/10 | Logical arc, good transitions, appropriate section lengths |
| Claim-evidence alignment | 6/10 | Primary claim overclaimed given confound; restricted analysis underreported |
| Visual communication | 5/10 | 4 of 6+ planned figures present; Sections 5.1-5.2 entirely text-heavy |
| Manuscript completeness | 5/10 | Broken cross-refs, missing appendix, unrendered Figure 2 |
| Prose quality | 8/10 | Evidence-forward, specific, no banned patterns, appropriate hedging |
| Title accuracy | 5/10 | Title features a failed theoretical framework |
