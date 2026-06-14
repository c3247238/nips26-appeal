# Writing Critique: Iteration 5

## Overall Assessment

The paper is well-written at the sentence level, with clear exposition of complex statistical methods and commendable transparency about negative results. The four-phase structure is logical, and the Introduction's three-weakness/four-contribution framing is effective. However, the paper has several major writing problems: numerical discrepancies between different sections and coefficient types, missing figures that leave two results sections without visual support, causal language that exceeds what the data supports, misframing of a negative result, and a confusing dual-rate taxonomy that will puzzle readers.

**Score: 6.5/10** (downgraded from the review's 7 due to additional issues identified below).

---

## Structural Coherence: 7/10

### Strengths

1. The three-weakness/four-contribution structure in the Introduction is tight. Each weakness has specific evidence (the r=-0.595 figure, five papers citing single-task limitation, absence of joint scaling map), and each contribution addresses a specific weakness. A reader knows exactly what the paper will do by paragraph 4.

2. The method-results pairing is clean: Section 3.1 matches 4.1, Section 3.2 matches 4.2, etc.

3. The Discussion (Section 5) is well-organized with subsections matching the contributions.

### Weaknesses

1. Section 4.3 ends with "Cross-Phase Transition," a self-referential bridge paragraph that breaks the results-reporting tone. A one-sentence transition would suffice.

2. The abstract (230 words) packs five quantitative claims with four p-values, three r-values, and two confidence intervals before the summary sentence. The information density is too high for an abstract -- front-load the take-home message.

3. Section 5.1 (The Causal Status of Absorption) is ~800 words with a 9-row Bradford Hill table. This depth is appropriate for a methods-contribution paper but excessive for what is primarily an empirical paper. The Bradford Hill table belongs in supplementary material.

4. The "Figures and Tables" listing after the Conclusion with raw filenames (e.g., `fig1_partial_correlations.pdf`) is a production artifact that must be removed.

---

## Claim-Evidence Integrity: 5/10

This is the weakest dimension. Multiple claim-evidence misalignments:

### 1. Coefficient type inconsistency (CRITICAL)

Table 3 labels paths a and b as "(std)" (standardized) but does not label c' and ab. The abstract uses c'=-0.003 for SCR (unstandardized, from -0.00290). Table 3 shows c'=-0.029 (standardized, from -0.0291). The Discussion (Section 5.1) uses c'=-0.003. A reviewer who checks Table 3 against the abstract will find a 10x discrepancy on the same quantity.

**Fix**: Label ALL columns in Table 3 as standardized. Ensure the abstract uses the same coefficient type. Add unstandardized values in supplementary material.

### 2. TPP Sobel z in summary files (CRITICAL)

final_results.json reports TPP Sobel z=2.63 (p=0.0085) from the reverse mediation path. The paper correctly uses z=2.08 (p=0.037) from the primary path. The paper itself is consistent, but any downstream consumer gets wrong numbers.

### 3. Taxonomy rate in summary files (CRITICAL)

final_results.json reports corrected_rate=0.923 (delta=0.0). The paper correctly uses 19.2%. The summary file was never updated after Phase 4 completed.

### 4. Phase 2 verdict framing

final_results.json: "PARTIALLY_SUPPORTED." The paper body (Section 4.2) honestly describes the negative result. The abstract says "exposes a critical limitation of the standard dominance-based metric." The verdict in the JSON summary does not match the paper's own description.

### 5. SP-F1 mediation classification

Table 3 footnote correctly classifies SP-F1 as "indirect-only" mediation (Zhao et al. 2010). But the paper title emphasis and Section 5.1 opening treat the suppression effect as the "central finding." A finding where the total effect is null (p=0.45) should not be the headline.

---

## Visual Communication: 4/10

### Missing figures

1. **Figure numbering jumps from 3 to 5** (no Figure 4). This is a formatting error that will confuse reviewers.

2. **Section 4.3 (gradient analysis)** has no figure. The text describes "443 points with gradient magnitude exceeding the 70th percentile threshold (0.69), with maximum gradient magnitude of 0.99" entirely in prose. A scaling surface paper without a gradient surface figure is incomplete.

3. **Section 4.4 (taxonomy correction)** has no figure. This is the only results section with zero visual elements. The 92.3% -> 19.2% correction is the most dramatic single number in the paper and deserves at least a comparison bar chart.

4. **No method overview figure**. A four-phase study with three different model-SAE combinations (48 Gemma Scope for Phase 1, GPT-2 Small for Phases 2+4, 420 SAEBench for Phase 3) needs a visual guide at the top of Section 3.

### What works

- All four included figures are properly forward-referenced.
- Figure captions are self-explanatory and include key takeaways.

---

## Notation and Terminology: 7/10

1. **"Absorption rate" vs "absorption score"**: The paper mostly uses "absorption rate" (R_abs) per the glossary, but Section 4.3 Setup says "absorption scores." Minor inconsistency.

2. **"Super-absorber"**: Used in Sections 4.2, 5.2, and 6 without formal definition or glossary entry. Needs definition on first use.

3. **Sobel test notation**: Consistent in the paper but not defined in notation.md. Minor.

4. **Bivariate r labeling**: Table 1 includes "Bivariate r (no covariates)" and "Partial r (no L0)" columns. The body text says "from r=-0.664 to r=-0.746" referring to partial-without-L0 and partial-with-L0, not the bivariate column. The column labels in Table 1 could be clearer: "Partial r (width+layer)" and "Partial r (width+layer+L0)."

---

## Causal Language Assessment

The paper's causal language exceeds what cross-sectional observational data can support:

1. "Absorption causally mediates L0's effect on SCR" -- should be "statistically mediates"
2. "Establishing absorption as the primary pathway" -- should be "suggesting absorption as a statistical pathway"
3. The Bradford Hill table (Table 7) imports epidemiological causation assessment to a setting where temporality (criterion 4) is assumed, not demonstrated
4. "The result reverses the prior concern" -- overstates; it shows the association survives L0 control, which is necessary but not sufficient for causation

The within-width matching null (Table 4) directly undermines the causal narrative. When width is held constant, there is no detectable quality difference between high- and low-absorption SAEs. This needs to be a prominent finding, not a Discussion caveat.

---

## Specific Prose Issues

1. **Passive voice overuse in Section 3**: "SAEs are partitioned into three width strata," "High-absorption and low-absorption SAEs are matched." The entire Method section is ~80% passive voice. Within ML norms but makes the prose monotone.

2. **Redundant Introduction/Conclusion**: The Conclusion restates the Introduction's contribution summary with the same numbers. The unique Conclusion content (within-width null caveat, practitioner recommendations) is buried. Tighten the repeated content.

3. **Section 5.2 length**: The "Why the Dominance Metric Fails" subsection is well-written but could be shorter. The Feature 8213 example and the "super-absorber" characterization could be a single paragraph instead of three.

4. **Dense abstract**: The first sentence runs 48 words. The second sentence packs three parenthetical methods with three numerical results. Consider splitting.

---

## What Works Well

1. **Transparency about negative results**: The paper is candid about within-width matching null, 0% cosine-calibrated rate, non-significant unlearning association, and the model fallback. Section 5.4 (Limitations) is thorough. This will earn reviewer trust.

2. **Table 3 footnote**: The indirect-only mediation classification with Zhao et al. (2010) citation is correctly handled. The dagger footnote explaining opposing-sign pathways is appropriately detailed.

3. **Pre-registered hypotheses in the paper**: Including H1-H4 with falsification criteria in Section 3.5 is good practice.

4. **No banned writing patterns detected**: No "In recent years," no "It is worth noting," no hype words. Every use of "significant" has a p-value.
