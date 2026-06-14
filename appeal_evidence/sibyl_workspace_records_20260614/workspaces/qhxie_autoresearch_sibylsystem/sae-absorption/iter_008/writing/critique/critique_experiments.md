# Section Critique: Experiments (Section 4)

**Section:** Section 4 -- Cross-Domain and Cross-Layer Absorption
**Reviewer:** sibyl-section-critic
**Date:** 2026-04-15

---

## Overall Score: 7 / 10

Section 4 presents the paper's primary empirical contribution -- layer-dependent and hierarchy-dependent absorption rates -- with clear tabular data, reasonable statistical analysis, and commendable transparency about the probe quality confound. The 15x layer variation finding is compelling and cleanly supported. However, the section suffers from several internal inconsistencies, a cross-reference mismatch on probe methodology, an inadequately titled subsection (4.4), and missed opportunities to connect results more precisely to the Method section. The issues range from minor wording problems to one substantive factual discrepancy that could erode reviewer trust.

---

## Strengths

### S1. Cleanly structured argumentation
The section follows the outline's strategic advice -- lead with the strongest, unconfounded finding (layer dependence), then introduce the confounded cross-domain result, then address the confound transparently. This ordering is effective and reviewer-friendly.

### S2. Statistical rigor
Bootstrap CIs, Kruskal-Wallis tests, paired permutation tests with Cohen's d, and explicit reporting of sample sizes are all present. The statistical apparatus exceeds the norms of the interpretability literature.

### S3. Transparent probe quality discussion (4.3)
The three-point enumeration of confounds (denominator asymmetry, missed absorption, spurious false negatives) is the section's strongest analytic contribution. Naming these explicitly preempts the most predictable reviewer objection.

### S4. Hedging decomposition (4.4)
Including the absorbed-vs-hedged breakdown across all eight configurations adds genuine analytic depth. The finding that hedging proportion increases with dictionary width and at later layers is a non-obvious, interpretable result.

---

## Weaknesses

### W1. CRITICAL -- Probe F1 inconsistency between Section 4 and Section 3 (and underlying data)

**Score impact: -1.5**

Section 4.1 states: "the sae_spelling pipeline (Chanin et al., 2024) achieves F1 >= 0.97 using in-context learning prompts" and "the sae_spelling ICL pipeline achieves F1 >= 0.97 at every layer."

Section 3.3 states: "the sae_spelling probe achieves F1 = 0.87 (L24) while the sklearn probe achieves F1 = 0.97 (L24). We use the higher-quality sklearn probes for absorption measurement."

The actual experimental data (`phase1_absorption_firstletter.json`) reveals that the first-letter absorption experiments used ICL-format probes at **position -6** achieving **F1 = 1.0 at all four layers** -- not F1 = 0.97 (the sklearn probe value) and not F1 = 0.87 (the sae_spelling LinearProbe value at L24).

This three-way mismatch (experiments section says 0.97, method section says sklearn probes were used, actual data says position-6 ICL probes with F1=1.0) is a factual error that will confuse readers and invite reviewer challenge. The claimed F1 >= 0.97 is actually a conservative understatement of F1 = 1.0, but the probe identity is wrong: Section 3.3 says sklearn probes were used, while the actual absorption experiment used a different probe (ICL format, position -6). This needs to be reconciled.

**Recommendation:** Clarify which probe was actually used for first-letter absorption measurement. If the ICL probe at position -6 with F1=1.0 was used (as the data shows), update both Sections 3.3 and 4.1 to state this consistently. The claim "F1 >= 0.97" should be either "F1 = 1.0" or qualified as "F1 >= 0.97 on the evaluation protocol."

### W2. Table 3 CI anomaly for city-country L24_16k

**Score impact: -0.5**

Table 3 reports city-country L24_16k with AR = 18.5% but 95% CI = [19.3, 42.2]. A point estimate of 18.5% falling outside its own 95% CI is mathematically impossible and indicates a data transcription error. The cross-domain summary data shows the absorption rate as 0.1850 with CI [0.193, 0.422] -- the CI appears to have been copied from a different metric (perhaps the loose absorption rate or an alternative estimator).

**Recommendation:** Verify the CI bounds for city-country L24_16k against the raw bootstrap output. If the CI is [0.119, 0.307] (from L24_65k, which looks plausible for that rate), the values may have been swapped between rows. Fix the table.

### W3. Section title mismatch -- 4.4 does not match the section header "Cross-Domain and Cross-Layer Absorption"

**Score impact: -0.5**

Section 4.4 ("Absorption-Hedging Decomposition by Hierarchy") analyzes the hedging decomposition for **first-letter only** across all eight SAE configs. Despite the subsection title saying "by Hierarchy," it does not actually decompose hedging across different hierarchy types -- it is purely a first-letter analysis. This subsection would more logically belong in Section 5 ("Causal Evidence and Hedging Decomposition"), as the outline places hedging decomposition there. Its presence in Section 4 disrupts the section's cross-domain narrative.

Additionally, the full hedging decomposition data (`hedging_decomposition_full_summary.md`) includes a city-language decomposition (66.7% strict hedging, n=3), which is not mentioned anywhere in Section 4.4. If cross-domain hedging data exists, it should be acknowledged even if sample sizes are too small for reliable conclusions.

**Recommendation:** Either (a) move Section 4.4 to Section 5 where the outline places it, or (b) rename the subsection to "First-Letter Absorption-Hedging Decomposition" and add a sentence noting that cross-domain hedging decomposition was attempted but yielded too few false negatives (n=3 for city-language) for reliable estimates.

### W4. "Single-L0 decomposition (Section 3.4)" -- incorrect cross-reference

**Score impact: -0.5**

Section 4.4 refers to "the single-L0 decomposition (Section 3.4)" but Section 3.4 describes the **absorption measurement pipeline** (false negative detection, integrated gradients, absorption detection). The hedging decomposition is described in **Section 3.6**. Furthermore, the term "single-L0 decomposition" does not appear in Section 3.6 or anywhere else in the paper; the hedging decomposition is defined as a three-category classification.

**Recommendation:** Change "(Section 3.4)" to "(Section 3.6)" and either define "single-L0 decomposition" or replace it with the terminology used in Section 3.6 ("three-category hedging decomposition").

### W5. Missing connection to the outline's planned Figure 4

**Score impact: -0.3**

The outline specifies Figure 4 as "Cross-Domain Absorption at L24" (a grouped bar chart by hierarchy with SAE width as hue). The actual Section 4.2 instead describes Figure 4 as a heatmap of absorption rate across hierarchy types, layers, and widths. This is not necessarily wrong (the heatmap may be a better visualization), but the figure description says RAVEL hierarchies are "measured at layer 24 only," which means most cells are missing -- a heatmap with predominantly empty cells is visually weak. The outline's original concept (a focused bar chart at L24 with CI) would more effectively communicate the cross-domain result.

**Recommendation:** Consider whether a grouped bar chart (as in the outline) or the heatmap better serves readers. If keeping the heatmap, acknowledge in the caption that most cells are empty due to single-layer RAVEL measurement.

### W6. Reversal claim (end of 4.2) lacks sufficient evidentiary support

**Score impact: -0.3**

Section 4.2 concludes: "This result reverses the pilot finding from layer 12, where semantic hierarchies appeared to show higher absorption than first-letter." This is an interesting observation, but the pilot L12 data is never shown in the paper (it exists only in internal pilot summaries). A reader has no way to verify the reversal claim. Citing internal pilot data without providing it undermines reproducibility.

**Recommendation:** Either (a) add a brief footnote or parenthetical with the L12 pilot rates for context, or (b) remove the reversal claim entirely and instead note that "layer-hierarchy interactions are non-trivial: absorption rankings across hierarchy types depend on which model layer is measured."

### W7. Inconsistent use of "15x" throughout the paper

**Score impact: -0.2**

The section header abstract says "varies 15x across model layers, from 2.2% at layer 18 to 34.5% at layer 24." The ratio 34.5/2.2 = 15.68, which rounds to 16x. The "15x" figure appears to be chosen for rounding convenience and is used consistently across all sections (intro, experiments, discussion, conclusion), so it is at least internally consistent. However, "approximately 16x" would be more accurate. This is minor.

### W8. Table 2 does not report the n_correct denominator clearly

**Score impact: -0.2**

Table 2 has a column labeled "$n_{\text{FN}}$ / Correct" which is ambiguous. At L24_16k, this reads "30 / 87" -- does this mean 30 false negatives out of 87 correctly classified tokens, or something else? The absorption rate formula in Section 3.4 defines AR as the fraction of *classes* with absorbed FNs, not the fraction of *tokens*. So 34.5% is 9/26 letters (approximately), not 30/87. The column label should clarify that "Correct" means the number of tokens correctly classified by the probe on raw activations, and that AR is computed at the class level.

**Recommendation:** Rename the column to "$n_{\text{FN}}$ / $n_{\text{correct}}$" and add a clarifying note that AR is computed at the class level (fraction of K classes with at least one absorbed FN), not the token level.

---

## Cross-Section Consistency Check

| Claim in Section 4 | Cross-reference | Status |
|---|---|---|
| "sae_spelling pipeline achieves F1 >= 0.97" | Method 3.3 says sklearn F1=0.97, sae_spelling F1=0.87; actual data shows ICL probe F1=1.0 | **MISMATCH** |
| "Kruskal-Wallis p = 0.005" | Intro: p=0.005; Discussion 8.1: p=0.005; Conclusion: p=0.005 | Consistent |
| "15x variation" | Intro: 15-fold; Discussion: 15x; Conclusion: 15x | Consistent |
| "rho = -0.756, p < 0.001" | Method 3.3: rho=-0.756, p<0.001; Discussion 8.3: rho=-0.756 | Consistent |
| Table 1 referenced "in Section 3" | Table 1 is in Section 3.2 | Consistent |
| "single-L0 decomposition (Section 3.4)" | Section 3.4 is absorption measurement, not hedging | **MISMATCH** |
| Hedging decomposition method | Should reference Section 3.6, not 3.4 | **MISMATCH** |
| L24 rates "25-35%" | Table 2: 25.5% and 34.5% | Consistent |
| City-country AR = 18.5%, CI [19.3, 42.2] | Point estimate outside CI | **DATA ERROR** |
| City-continent "n = 200" in confound discussion | Cross-domain data: 200 cities for continent | Consistent |
| Probe F1 values in Table 3 | Match probe_training_full.json (0.843, 0.789, 0.823) | Consistent |

---

## Actionable Revisions (Priority Order)

1. **[HIGH]** Fix the probe identity discrepancy (W1). Determine which probe was actually used for first-letter absorption and make Sections 3.3, 4.1, and Table 2 consistent.
2. **[HIGH]** Fix the CI anomaly in Table 3 for city-country L24_16k (W2).
3. **[MEDIUM]** Correct the cross-reference from "Section 3.4" to "Section 3.6" and fix the terminology (W4).
4. **[MEDIUM]** Relocate or retitle Section 4.4 (W3).
5. **[LOW]** Clarify the Table 2 column header (W8).
6. **[LOW]** Substantiate or soften the reversal claim in 4.2 (W6).
7. **[LOW]** Reconsider Figure 4 format (W5).

---

## Summary Judgment

The Experiments section is **structurally sound and above average for the interpretability literature** in its statistical reporting and transparency about confounds. The layer dependence finding (Section 4.1) is the paper's strongest result and is presented effectively. The cross-domain analysis (4.2) and confound discussion (4.3) are honest and well-organized.

However, the probe identity discrepancy (W1) is a substantive factual error that must be fixed before submission -- a reviewer who cross-checks Sections 3 and 4 will notice the contradiction. The CI anomaly in Table 3 (W2) is an obvious mathematical impossibility that signals carelessness. The misplaced cross-reference (W4) and awkward positioning of Section 4.4 (W3) are structural issues that a careful revision can resolve quickly.

After addressing W1 and W2, this section would score 8/10. After addressing all issues, 8.5/10.
