# Critique: Method (Section 3)

## Summary Assessment
The method section is well-structured and covers the key experimental protocols systematically: feature hierarchies, probe training, absorption measurement, activation patching, benign/pathological diagnostic, hedging decomposition, and architecture comparison. The statistical reporting is commendably thorough. However, several numerical inconsistencies between the method section and actual experimental data undermine trust, and the section omits key details about what was actually run (e.g., RAVEL hierarchies measured only at L24, not all four layers). The prose is clear but occasionally too dense for a methodology section -- some procedural detail belongs in an appendix.

## Score: 6/10
**Justification**: The section provides a complete experimental framework with honest quality caveats and well-defined controls. It loses points for: (1) a critical numerical conflict between the outline and the actual data on first-letter absorption, (2) the F1=1.0 claim conflating two different probe training pipelines, (3) missing explanation of why RAVEL hierarchies are measured only at L24, and (4) no explicit connection between the method and what was actually measured in the experiments section. Reaching a 7+ requires resolving all numerical inconsistencies and clarifying the scope of each experimental component.

## Critical Issues

### Issue 1: Outline-Method-Experiments Inconsistency on First-Letter Absorption Rate
- **Location**: Section 3.2, line 7; cross-reference with outline Section 4 and experiments Section 4.2
- **Quote**: "probes achieve perfect F1 = 1.0 at all layers (trained at token position -6 following the sae-spelling protocol)"
- **Problem**: The outline (Section 4) states first-letter absorption is 42.9% at L24, while the experiments section and consolidation data report 27.1% at L24_16k. The method section references probes with F1=1.0, which is correct for the sae-spelling pipeline (position -6, confirmed by `absorption_firstletter.json`). However, the consolidation summary reports first-letter F1=0.97, referencing the generic probe training pipeline (position -2, per `probe_training.json` where F1_weighted=0.960). The method section must explicitly clarify that TWO distinct probe training pipelines exist for first-letter: the sae-spelling pipeline (position -6, F1=1.0) and the generic sklearn pipeline (position -2, F1=0.96). Without this, a reviewer will be confused by conflicting F1 values across sections.
- **Fix**: Add a sentence in Section 3.2 stating: "Note: the generic probe training pipeline (Section 3.2, position -2) also trains first-letter probes (F1=0.96 at L24), but all first-letter absorption measurements use the sae-spelling pipeline (position -6, F1=1.0) for maximum reliability." Also flag to the outline author that the 42.9% first-letter rate in the outline is stale -- actual data shows 27.1%.

### Issue 2: Missing RAVEL Layer Scope Justification
- **Location**: Section 3.1 and 3.2 (entire section)
- **Quote**: "We train linear probes [...] at layers l in {6, 12, 18, 24}"
- **Problem**: The method section implies that all four hierarchies are measured at all four layers. In reality, RAVEL hierarchies (city-continent, city-language, city-country) are measured only at L24 (experiments section 4.2: "RAVEL hierarchies are measured only at L24 because RAVEL probes achieve their best F1 at this layer"). The method section never states this constraint. A reviewer reading only the method would expect cross-domain absorption results at all four layers, then be confused when Section 4 presents RAVEL results only at L24.
- **Fix**: Add a scope statement at the end of Section 3.2 or beginning of 3.3: "Because RAVEL probes achieve adequate quality (F1 >= 0.73) only at L24 -- with F1 dropping to 0.37-0.72 at earlier layers -- cross-domain absorption is measured at L24 only. First-letter absorption is measured at all four layers, leveraging its F1=1.0 probes across all layers."

## Major Issues

### Issue 3: Activation Patching Section Does Not Mention the Cross-Domain Failure
- **Location**: Section 3.4, lines 53-67
- **Problem**: The activation patching protocol is described generically, applicable to any hierarchy. But the experiments show this method completely failed for cross-domain (city-continent): child-zeroed recovery was 0.05% vs. 14.5% for control (reversed effect). The method section should at least mention that the protocol was applied to both first-letter and cross-domain, with different results. Currently, the method implies activation patching works uniformly across all hierarchies.
- **Fix**: Add a sentence in the "Scope" paragraph: "We apply this protocol to first-letter (concentrated hierarchy) and city-continent (distributed hierarchy) to test whether the causal mechanism generalizes across hierarchy types."

### Issue 4: Architecture Comparison Uses Kruskal-Wallis, Not ANOVA
- **Location**: Section 3.7, line 102
- **Quote**: "The analysis uses a two-factor design: architecture x hierarchy, with absorption rate as the dependent variable."
- **Problem**: The method says it uses a "two-factor design" but the statistics reported in the consolidation summary use Kruskal-Wallis tests (non-parametric), not a two-factor ANOVA. The discussion section refers to "ANOVA p=0.50-0.75" while the data says "Kruskal-Wallis p=0.754" at L12. The method section should specify which test is used and why (e.g., non-normality of absorption rates). Using "two-factor design" without specifying the test creates ambiguity.
- **Fix**: Replace "The analysis uses a two-factor design" with "We test the architecture factor and hierarchy factor separately using Kruskal-Wallis tests (non-parametric, chosen because absorption rates are not normally distributed). Two-way interaction is assessed by comparing effect sizes."

### Issue 5: Benign/Pathological Diagnostic Scope is Very Narrow
- **Location**: Section 3.5, line 82
- **Quote**: "Scope. 50 city-continent entities, 30 contexts each, yielding 1,471 false negative instances."
- **Problem**: The diagnostic is applied ONLY to city-continent. The method section does not explain why only one hierarchy was tested. Given the paper's central finding that absorption is hierarchy-specific, testing pathological status on only one hierarchy (city-continent, which has moderate probe quality F1=0.87) weakens the claim "absorption is ALWAYS pathological." A reviewer will ask: is absorption also 100% pathological for first-letter? For city-country? The method should justify the single-hierarchy scope or acknowledge it as a limitation.
- **Fix**: Add justification: "We conduct this diagnostic on city-continent (F1=0.87, the highest RAVEL probe quality) as the primary test. First-letter is excluded because activation patching shows a different (concentrated) mechanism, and city-country's below-gate probe quality (F1=0.73) would confound the diagnostic."

### Issue 6: "Approximately 500 child entities" for First-Letter is Imprecise
- **Location**: Section 3.1, line 7
- **Quote**: "approximately 500 child entities (common English words)"
- **Problem**: The actual data shows n_test_words=500 and n_probe_train_words=1033 (from `absorption_firstletter.json`). The method should use the exact number. "Approximately 500" is vague for a methodology section. Also, the probe training data shows 4,132 training examples (1,033 words x ~4 prompts), not "approximately 500."
- **Fix**: Replace "approximately 500 child entities" with "500 test words (1,033 total including training set)." The 4,132 training examples mentioned later in Section 3.2 are consistent.

## Minor Issues

- **Line 9, city-continent**: "1,567 child entities" -- the probe training data shows n_train=1,632 and n_test=409, totaling 2,041 examples but likely 1,567 unique entities (with some appearing in both train/test). Clarify whether 1,567 is entities or examples.
- **Line 18, first-letter token position**: "token position -6 (the position encoding the first character of the next word)" -- this phrasing is somewhat misleading. Position -6 encodes information ABOUT the first character, but it is the token 6 positions before the target, not the position OF the first character. Rephrase for precision.
- **Line 21, regularization grid**: The grid includes C=10.0, but the probe data shows best_C=10.0 for city-country at all layers, which risks overfitting (training accuracy=1.0 for all city-country probes). This is not acknowledged.
- **Line 27, city-country F1=0.73**: The text says "below gate" but does not state the actual threshold for "below gate." The definition (F1 < 0.80) should be stated inline or referenced.
- **Line 31, Table 1 caption**: References "figures/table1_probe_quality.pdf" -- verify this file exists and matches the data in probe_training.json.
- **Line 42, integrated-gradients attribution**: No details on the number of interpolation steps, baseline choice, or implementation. These should be specified or directed to an appendix.
- **Line 44, SAE configurations**: "Matryoshka (m=32,768)" -- the method states this is from SAEBench, but does not explain that Matryoshka uses a nested dictionary structure that makes the effective width different from the nominal width. This should be noted.
- **Line 60, reconstruction formula**: $\hat{\mathbf{x}}^{(c \to 0)} = \mathbf{W}_{\text{dec}} \mathbf{z}^{(c \to 0)} + \mathbf{b}_{\text{dec}}$ -- consistent with notation.md. Good.
- **Line 94, multi-L0 analysis**: References "8 Gemma Scope JumpReLU configurations at layers 6, 12, 18, 24 with two dictionary widths each." This is 4 layers x 2 widths = 8 configs, which is correct. But the L0 range "22 to 176" should cite the source of these L0 values.
- **Line 100, architecture comparison at L24**: States "at layer 24, only JumpReLU (both widths) and Matryoshka are available" -- but Section 3.7 says "3 architectures" (JumpReLU 16k, JumpReLU 65k, Matryoshka 32k). Counting JumpReLU at two widths as two architectures is debatable. The consolidation summary counts these as 3 architectures at L24, which is consistent but should be made explicit (e.g., "three SAE configurations from two architecture families").
- **Line 106, "single GPU with >= 24 GB VRAM"**: The experiments actually ran on specific hardware. State what GPU was used, not just the minimum requirement.
- **Lines 47-48, random direction baseline**: The section describes replacing w_p with a random unit vector and "re-measuring the false negative rate." This is not the same as the absorption rate. Clarify: does this measure the FN rate on raw activations (to check if a random direction also has FNs) or on SAE reconstructions?

## Visual Element Assessment
- [x] Figures/tables match outline plan -- Table 1 (probe quality) is planned and referenced
- [x] All visuals referenced before appearance -- Table 1 referenced in line 29 before the figure tag
- [ ] Captions are self-explanatory -- Table 1 caption includes gate annotations but does not explain what the gate symbols mean (checkmark/tilde/cross are defined in text but not in caption)
- [ ] No text-heavy sections that need visual support -- Section 3.3 describes the absorption pipeline procedurally; a flow diagram would help readers follow the measurement steps (raw activation -> probe -> SAE encoding -> probe -> compare). The outline plans a schematic in Figure 1 (left panel), but the method section should reference it.

## What Works Well
- **Honest quality gates (Section 3.2)**: The three-tier quality gate system (strict/relaxed/below) with explicit acknowledgment that city-country is "below gate" is exactly the kind of transparent reporting that builds reviewer trust. The phrase "absorption rates should be treated as upper bounds" for relaxed-pass probes is a model of honest scientific communication.
- **Complete control battery (Section 3.3)**: Three distinct baselines (random direction, shuffled hierarchy, probe-only) provide strong defense against "this is just SAE reconstruction error" objections. Each control targets a different confound, and the statistical framework (bootstrap CI, Kruskal-Wallis with Bonferroni) is appropriate for the data structure.
- **Hedging decomposition (Section 3.6)**: The strict/compensatory/persistent three-way classification is a genuine methodological contribution. Decomposing false negatives by whether the main parent feature fires resolves the ambiguity in prior work's hedging metric and enables the multi-L0 analysis that reveals the mechanism's sensitivity to sparsity budget.
