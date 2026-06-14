# Critique: Method

## Summary Assessment
The Method section presents a well-structured theoretical framework and experimental procedures for studying SAE feature absorption through the lens of critical phenomena. The phase transition analogy is clearly articulated, and the six experimental sub-procedures are logically organized. However, there is a critical inconsistency between the described CV analysis parameters and what the Experiments section actually reports, as well as missing coverage of two experiments that appear in the outline.

## Score: 6/10
**Justification**: The section is technically sound on the core physics framework and correctly describes the first six experiments (H1-H6). However, it suffers from: (1) a critical mismatch between the CV analysis sparsity value in the text (5×10⁻⁵) and what experiments.md actually reports (0.001), creating doubt about what was actually run; (2) missing sections for the activation patching and steering effectiveness experiments that appear in the outline and experiments; (3) incomplete coverage of the cross-layer measurement at λ_c (Phase 4 in proposal), which remains "pending" but is a key planned contribution. Fixing these would bring the score to 8+.

## Critical Issues

### Issue 1: CV Analysis Sparsity Value Inconsistency
- **Location**: Section 3.7, "CV Analysis (H4: Variance Paradox)", step 2
- **Quote**: "Compute per-feature CV across 1000 samples at λ = 5 × 10⁻⁵"
- **Problem**: The Experiments section 4.4 explicitly states the CV analysis was conducted at λ = 0.001 ("across n = 1000 samples at λ = 0.001"), not λ = 5×10⁻⁵ as written in the method. This is a direct contradiction that suggests either the method was written from the proposal before execution, or the experiments deviated from the plan without updating the method.
- **Fix**: Either correct the method to λ = 0.001 (if experiments is authoritative) or verify the actual experimental logs to confirm the true sparsity value used.

### Issue 2: Missing Experiments from Outline
- **Location**: Section 3.7 (Experimental Procedure); compare against outline.md and experiments.md
- **Quote**: N/A - experiments 4.7 (Activation Patching Validation) and 4.8 (Steering Effectiveness by CV) are completely absent from the method section
- **Problem**: The outline's Figure & Table Plan references these experiments, and experiments.md describes them in detail (recovery percentages 48.8-75.2%, steering effect difference of +0.078). The method section stops at H6 (Graph Topology), leaving readers without procedural understanding of these two additional experiments. This breaks the completeness dimension.
- **Fix**: Add sections 3.9 (Activation Patching Validation) and 3.10 (Steering Effectiveness by CV) describing methodology for these experiments, matching the level of detail in the other experimental sub-sections.

### Issue 3: Cross-Layer Measurement at Critical Sparsity Remains Unresolved
- **Location**: Section 3.7, "Cross-Layer Measurement (H3: Layer Criticality)", step 2; also proposal.md Phase 4
- **Quote**: "Test whether layers exhibit heterogeneous absorption (versus uniform saturation)"
- **Problem**: The method describes cross-layer measurement at λ = 10⁻³ (standard sparsity), which the experiments confirm yields uniform saturation. The proposal (Phase 4) explicitly proposes cross-layer measurement at λ_c = 5×10⁻⁵ to test layer heterogeneity at the true critical point. This measurement is described as "pending" in the experiments section, yet the method provides no protocol for it. The cross-layer heterogeneity question remains unanswered because the wrong sparsity was tested.
- **Fix**: Add a subsection describing the planned cross-layer measurement at λ_c = 5×10⁻⁵, or explicitly note that this measurement is deferred to future work and explain why the method was only executed at λ = 10⁻³.

## Major Issues

### Issue 4: Terminology Inconsistency with Notation.md
- **Location**: Section 3.3, "Order Parameter and Control Parameter"
- **Quote**: "Let m(λ) denote the order parameter: mean absorption rate"
- **Problem**: The notation.md standard (Table, "Absorption Metrics" section) defines absorption rate as α, not m. The symbol m is reserved for the order parameter in the phase transition analogy (magnetization). The method uses m(λ) for absorption rate but notation.md uses α for this same quantity. This creates confusion about whether m(λ) is the absorption rate or the magnetization analog.
- **Fix**: Either (a) use α consistently for absorption rate as notation.md specifies, or (b) clarify that m(λ) is the order parameter (absorption rate as analogy to magnetization) and note that α = m(λ) in this context. Option (b) is preferred to maintain the physics analogy.

### Issue 5: Absorption Rate Notation Variance
- **Location**: Section 3.7, step 1 in Sparsity Sweep; also compare across sections
- **Quote**: "Compute absorption rate m(λ)" in method vs "absorption rate α" in experiments.md Table 3
- **Problem**: The experiments section uses α for absorption rate (Table 3: "Absorption Rate" column with values 1.0), while the method uses m(λ). This is inconsistent across sections.
- **Fix**: Establish α as the standard symbol for absorption rate throughout, with m reserved for the order parameter/magnetization analogy.

### Issue 6: Co-occurrence Formula Description Lacks Operability
- **Location**: Section 3.5, "Co-occurrence Formula"
- **Quote**: "We test a revised co-occurrence formula for predicting absorption"
- **Problem**: The formula S_revised is presented but the operationalization is unclear: How is the revised score computed from the raw data? What threshold classifies a pair as exhibiting co-occurrence? The description says "We correlate S_revised with absorption rate" but does not specify the regression or correlation method (Pearson? Spearman?).
- **Fix**: Add a sentence specifying: "We compute Pearson correlation r between S_revised scores and absorption rates A_j across all feature pairs, reporting r and associated p-values."

## Minor Issues

### Minor 1: "Training-Free" vs "Training-Free Detector" terminology
- **Location**: Section 3.2
- **Issue**: The section refers to "training-free absorption detector" but the outline and glossary prefer "training-free detector" or the specific metric name A_j. The phrase "training-free absorption detector" is repeated.
- **Fix**: Use "the A_j training-free detector" or simply "the A_j metric" on first use, then abbreviate to "A_j" thereafter.

### Minor 2: Section numbering doesn't match outline
- **Location**: Entire section
- **Issue**: The outline uses "3.1", "3.2" etc. for main subsections, but the method section uses "3.1", "3.2" etc. which actually aligns. However, experiments 4.7 and 4.8 have no corresponding method subsections.
- **Fix**: Ensure method section numbering directly supports the experiments it precedes, or note that additional experiments (4.7, 4.8) are described elsewhere.

### Minor 3: Figure references precede figure appearances
- **Location**: The method section plans figures in the outline/figure comments but has no in-text figure references
- **Issue**: The outline specifies figures should be referenced BEFORE they appear. The method section currently has no in-text figure references at all (the <!-- FIGURES --> block is a comment, not rendered text).
- **Fix**: Add sentences like "Figure 1 illustrates the quasi-critical phase transition behavior" at the appropriate points in the descriptions of each experiment sub-section.

### Minor 4: Statistical methods section is thin
- **Location**: Section 3.8, "Statistical Methods"
- **Issue**: Mentions t-test, Pearson correlation, and R² but does not specify thresholds (e.g., p < 0.01 for significance), exact formulations, or software/packages used.
- **Fix**: Expand to: "We employ t-test with Welch's correction for unequal variances (scipy.stats.ttest_ind), Pearson correlation with associated p-values (scipy.stats.pearsonr), and R² from numpy.polyfit for scaling collapse quality. All tests use significance level α = 0.01."

### Minor 5: Graph topology λ value discrepancy
- **Location**: Section 3.7, step 1 in Graph Topology Analysis
- **Quote**: "Construct absorption graphs at each layer at λ = 10⁻³"
- **Problem**: The experiments section (Table 5) shows this was executed at λ = 0.001, which is consistent. However, the method does not connect this choice to the fact that λ = 10⁻³ is past the critical point for all layers (as shown in H3 results).
- **Fix**: Add one sentence: "We use λ = 10⁻³ since this sparsity exceeds λ_c for all layers (per H3 results), ensuring comprehensive edge coverage in the absorption graph."

## Visual Element Assessment
- [ ] Figures/tables match outline plan: PARTIAL - method references planned figures but has no in-text references to figures
- [ ] All visuals referenced before appearance: NO - the method section does not reference any figures in the body text
- [ ] Captions are self-explanatory: N/A - figures are in separate generation scripts, not in the method section
- [ ] No text-heavy sections that need visual support: The phase transition framework (Section 3.3) would benefit from a small schematic diagram showing the order parameter m(λ) curve, control parameter λ, and susceptibility peak χ

## What Works Well

1. **Phase Transition Framework Clarity (Section 3.3)**: The analogy to statistical physics is clearly articulated with proper definitions of order parameter, control parameter, critical point, susceptibility, and finite-size scaling. The quasi-critical framing (χ_ratio = 1.88 < 3.0) is well-explained and distinguishes this work from a true sharp phase transition.

2. ** SAE Architecture Description (Section 3.1)**: The mathematical formulation is correct, the hook name is precise (`blocks.6.hook_resid_pre`), and the use of SAELens pretrained SAEs is clearly stated. The dictionary sizes {6144, 12288, 24576} are correctly identified.

3. **Logical Section Organization**: The progression from SAE basics → absorption definition → phase transition physics → analysis methods → experimental procedure is coherent and follows a natural reading order. A reader new to the topic can follow the intellectual thread.

4. **Explicit Connection to Prior Work**: The method correctly cites Chanin et al. (2024) for the A_j formula and notes that layer 6 was identified as an absorption hotspot "based on prior work", demonstrating proper scholarly attribution.
