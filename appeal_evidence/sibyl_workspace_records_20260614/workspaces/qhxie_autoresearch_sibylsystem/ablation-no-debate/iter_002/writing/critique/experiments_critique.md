# Critique: Experiments

## Summary Assessment
The Experiments section presents a well-structured, comprehensive empirical evaluation with seven sub-experiments covering factorial decomposition, robustness validation, intervention tests, and structural ablations. The section honestly reports negative results and includes rich statistical detail. However, it suffers from critical inconsistencies between reported numbers in different parts of the paper, unclear metric definitions, and a logical gap in how the factorial results connect to the broader claims. The writing is generally clear but contains several instances of imprecise statistical language and one potentially misleading figure reference.

## Score: 6/10
**Justification**: The section has strong bones -- comprehensive coverage, honest negative results, good visual communication -- but is undermined by numerical inconsistencies (different absorption values for the same conditions across tables/paragraphs), ambiguous metric definitions, and a failure to fully reconcile the factorial findings with the multi-seed validation numbers. To reach 7-8, it needs: (1) consistent numbers throughout, (2) explicit clarification of which metric is used where, (3) explanation of why Condition D absorption (0.436) differs from the "trained SAE" absorption (0.477) in H1, and (4) removal of statistical overclaiming.

## Critical Issues

### Issue 1: Inconsistent Absorption Values for Identical Conditions
- **Location**: Table 2 (line 25-30) vs. H1 paragraph (line 38) vs. Table 1 row "H_Mech factorial" (line 7)
- **Quote**: Table 2 shows Condition D (trained/trained) absorption = 0.436 +/- 0.043; H1 paragraph states trained SAEs show absorption 0.477 +/- 0.022; Table 1 states "Encoder effect 0.843 +/- 0.082"
- **Problem**: These are supposedly measuring the same or closely related quantities but yield different values with no explanation. Condition D is "full training" -- why does it show 0.436 while H1's "trained SAEs" show 0.477? Are these different metrics (overlap vs. Jaccard)? Different experimental conditions? The reader cannot tell.
- **Fix**: Add an explicit footnote or sentence explaining that the factorial decomposition (Section 6.1) uses the overlap-fraction metric averaged across all L0 levels and hierarchy strengths, while H1 uses the Jaccard overlap metric at L0=32 and similarity=0.67. Better yet, use the same metric throughout or explicitly state when metrics differ and why.

### Issue 2: Metric Inconsistency Not Addressed in the Section
- **Location**: Section 6.1, paragraph 1 (line 17-19) and Section 6.2 (line 36-38)
- **Quote**: "The encoder effect E_enc = alpha(B) - alpha(A) isolates the contribution of encoder alignment" vs. the H1 metric described as "Jaccard overlap"
- **Problem**: The Discussion section (Section 7.5, third limitation) admits that "three different absorption metrics (cosine similarity, overlap fraction, Jaccard index) are used across experiments without establishing formal equivalence." But this critical caveat is buried in the Limitations subsection and never mentioned in the Experiments section where the numbers are actually reported. A reader evaluating the empirical claims has no way to know that Table 2's 0.861 and H1's 0.477 may not be directly comparable.
- **Fix**: Add a paragraph at the start of the Experiments section (before 6.1) explicitly stating which metric is used for each experiment and noting that numerical scales differ across metrics. Alternatively, standardize on one metric for all experiments.

### Issue 3: Post-Hoc Criterion Revision Undermines Confirmatory Claims
- **Location**: Section 6.1, paragraph 3 (line 32)
- **Quote**: "The original pass criteria (B approx D and C approx A) failed at a 6.7% rate (1/15) because Condition D consistently shows lower absorption than Condition B -- a decoder disentanglement effect we discuss in Section 7.1. Under revised criteria (encoder effect > 0.5, decoder effect < 0.1), all 15 runs pass."
- **Problem**: The paper presents H_Mech as "Confirmed" in Table 1 and the Abstract, but the original pre-registered criteria failed on 93.3% of runs. The revised criteria were developed after observing the data. This is exploratory validation masquerading as confirmatory hypothesis testing -- a form of p-hacking that top venues flag as a serious methodological concern.
- **Fix**: Either (a) reframe H_Mech as "exploratory" rather than "confirmed," or (b) pre-register the revised criteria on a held-out validation set and report both. The current presentation is misleading.

## Major Issues

### Issue 4: Missing Baseline Comparison for H_Safe
- **Location**: Section 6.4 (line 54-62)
- **Quote**: "Safety-critical features show mean absorption of 0.967 +/- 0.010; non-safety features show 0.968 +/- 0.013."
- **Problem**: Both groups show near-complete absorption (~97%). The null result is clear, but there is no comparison to the synthetic data absorption rates (~48-55%). Are real SAEs expected to show much higher absorption? Is the Gemma Scope SAE architecture comparable to the synthetic experiments? Without this context, the reader cannot assess whether 97% absorption is expected or anomalously high.
- **Fix**: Add a sentence comparing real-SAE absorption to synthetic-SAE absorption, or cite prior work (e.g., Chanin et al., 2024) reporting comparable rates on real LLM SAEs.

### Issue 5: Steering Protocol Mismatch Between Method and Experiments
- **Location**: Method Section 5.5 vs. Experiments Section 6.3
- **Method quote**: "We steer feature activations toward parent directions by adding a scaled direction vector: f_steered = f_baseline + alpha * v_parent"
- **Experiments quote**: "We identify absorbed features (those with alpha_abs > 0.3 and parent-child overlap > 0.5) and non-absorbed features (those with alpha_abs < 0.1)"
- **Problem**: The Method section describes steering on synthetic SAE features with known parent-child structure, but the Experiments section does not clarify whether the steering was done on synthetic or real SAEs. Given that the pilot summary reports H3 as "pilot completed" with ratio 0.97x, and the Method describes a synthetic protocol, the experiment likely used synthetic data -- but this is never explicitly stated.
- **Fix**: Add one sentence at the start of Section 6.3 stating "All steering experiments use the synthetic hierarchy setup described in Section 5.1."

### Issue 6: Vague Statistical Language
- **Location**: Section 6.2, line 42
- **Quote**: "The effect size is extreme (Cohen's d > 10), indicating that trained-versus-random classification is essentially deterministic given our measurement protocol."
- **Problem**: "Essentially deterministic" is an overstatement. Cohen's d > 10 indicates a very large effect, but "deterministic" implies zero variance or perfect classification accuracy, which is not what Cohen's d measures. This is hype language that weakens the section's credibility.
- **Fix**: Replace with "Cohen's d > 10 indicates near-complete separation between trained and random conditions, with minimal overlap in distributions."

### Issue 7: Figure 1 Caption Mismatch
- **Location**: Figure 1 caption (line 21)
- **Quote**: "Condition B (trained encoder + random decoder) shows absorption comparable to full training, while Condition C (random encoder + trained decoder) remains at baseline."
- **Problem**: Table 2 shows Condition B = 0.861 and Condition D = 0.436. These are not "comparable" -- Condition B is nearly 2x Condition D. The caption's claim reflects the original (failed) criteria where B approx D was expected, but the actual data show B >> D. The caption should reflect the actual finding: trained encoder alone produces HIGHER absorption than full training.
- **Fix**: Update caption to: "Condition B (trained encoder + random decoder) shows higher absorption than full training (Condition D), revealing a decoder disentanglement effect. Condition C (random encoder + trained decoder) remains at baseline."

## Minor Issues

- **Section 6.1, line 19**: "Condition D (full training, 0.436 +/- 0.043 on the overlap metric, but the encoder effect itself is 0.843)" -- The parenthetical about the encoder effect is a non-sequitur in this sentence. Move to the next sentence or clarify the connection.
- **Section 6.3, line 46**: "For each of 5 seeds, we identify absorbed features... and non-absorbed features, then steer both groups toward parent directions at alpha values {0.5, 1.0, 2.0, 5.0}." -- The Method section (5.5) lists alpha values as {0.5, 1.0, 2.0, 5.0} but the primary metric is at alpha = 2.0. Why report all four alphas if only one is used for the primary test? Clarify or justify.
- **Section 6.4, line 56**: "Using the GPT-2 Small residual SAE (layer 8, d_sae = 24576) via SAELens" -- The Method section (5.6) says "gemma-scope-2b-pt-res... targeting layer 12 with d_sae = 16384." Which is correct? GPT-2 layer 8 / d_sae=24576 or Gemma layer 12 / d_sae=16384? This is a direct contradiction.
- **Section 6.5, line 66**: "A one-way ANOVA confirms the effect: F = 4718.81, p < 1e-10." -- With only 3 levels (0.5, 0.67, 0.8), a one-way ANOVA is appropriate but reporting F = 4718.81 with only 3 groups suggests extremely large sample sizes. Clarify the degrees of freedom or use a more informative test (linear trend analysis).
- **Section 6.6, line 80**: "A one-way ANOVA confirms the effect: F = 4342.17, p < 1e-10." -- Same issue as above. With 3 L0 levels, the F-statistic is driven by sample size, not effect magnitude. Report eta-squared or partial eta-squared for effect size interpretation.
- **Table 1, line 9**: "H3 steering | Negative result | Sensitivity ratio 0.915 +/- 0.396 | p = 0.433" -- The experiments section reports 0.776 +/- 0.066 (p = 0.273) at alpha=2.0, but Table 1 shows 0.915 +/- 0.396 with p = 0.433. Which is the primary result? The inconsistency between Table 1 and Section 6.3 undermines reproducibility.
- **Section 6.7, line 94**: "The Pearson correlation between seed-level train and test means is r = 0.998 (p = 1.44e-04)." -- With only 5 seeds, r = 0.998 is suspiciously high. Report the 95% confidence interval for the correlation to show uncertainty bounds.

## Visual Element Assessment
- [x] Figures/tables match outline plan -- All 7 figures and 3 tables from the outline are present or referenced.
- [x] All visuals referenced before appearance -- Figures and tables are referenced in text before their inclusion.
- [ ] Captions are self-explanatory -- Figure 1 caption contains a misleading claim (see Critical Issue 7). Table 2 lacks a caption entirely.
- [ ] No text-heavy sections that need visual support -- Section 6.3 (steering) and 6.4 (safety) are text-heavy with minimal numerical detail. A small summary table for each would improve clarity.

## What Works Well
1. **Honest negative results**: Sections 6.3 and 6.4 explicitly label steering and safety as negative/null results, with full statistical reporting. This is scientifically admirable and strengthens the paper's credibility.
2. **Comprehensive statistical reporting**: Every claim includes a specific test statistic, p-value, and effect size (where applicable). The statistical detail is appropriate for a top venue.
3. **Clear experimental structure**: The seven subsections follow a logical progression from central claim (factorial) to robustness (multi-seed) to interventions (steering, safety) to structural characterizations (hierarchy strength, sparsity, generalization). This structure makes the section easy to navigate.
