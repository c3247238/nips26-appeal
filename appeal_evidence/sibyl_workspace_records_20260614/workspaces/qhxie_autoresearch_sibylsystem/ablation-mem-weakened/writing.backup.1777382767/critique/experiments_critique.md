# Critique: Experiments Section

## Summary Assessment

The Experiments section presents a clear, well-structured reporting of null results with good use of tables and figure references. The data reporting is generally accurate and the narrative effectively communicates the core finding: no significant correlation between absorption rate and downstream task performance. However, the section suffers from several figure-caption mismatches, a critical statistical notation error, missing controls discussion, and incomplete visual element coverage relative to the outline plan.

## Score: 6/10

**Justification**: The section successfully reports results with accurate numbers and reasonable structure, earning a solid base score. It loses points for: (1) a critical error in H3 CV reporting (negative CV is impossible), (2) figure/caption mismatches that confuse the reader, (3) missing discussion of controls promised in the Method section, and (4) prose-heavy passages where visual elements would improve clarity. To reach 7-8, fix the CV error, reconcile figure references, add controls results, and tighten the prose. To reach 9+, add effect size discussion and address the power analysis more directly.

---

## Critical Issues

### Issue 1: Impossible Negative Coefficient of Variation in H3

- **Location**: Section 5.4, H3 paragraph, Table 1
- **Quote**: "the coefficient of variation CV = sigma / |mu| = 0.932 exceeds the 0.5 threshold" and in Table 1: "CV = 0.932 (H2)"
- **Problem**: The coefficient of variation (CV) is defined as the ratio of standard deviation to the *mean* (not absolute mean). With H2 slopes of beta_4 = -0.010 and beta_8 = -0.286, the mean is negative (mu = -0.148), so CV = sigma / |mu| = 0.148 / 0.148 = 1.0 (approximately), which is positive. However, the source data in `correlation_report_full.json` reports `cv_h2 = -0.9315` and `cv_h1 = -1.0789`. These negative values suggest the code computed CV = sigma / mu (without absolute value on the denominator), which is incorrect. The section text says "CV = sigma / |mu|" but uses the negative value from the JSON, creating an internal contradiction. A negative CV is mathematically impossible by the stated definition.
- **Fix**: Recalculate CV correctly using CV = sigma / |mu|. For H1: mu = (-0.630 + 0.024)/2 = -0.303, sigma = 0.462, CV = 0.462 / 0.303 = 1.52. For H2: mu = (-0.286 + (-0.010))/2 = -0.148, sigma = 0.138, CV = 0.138 / 0.148 = 0.93. Report the correct positive values and update Table 1. Also fix the source JSON if possible.

### Issue 2: Figure-Caption Mismatches and Incorrect Figure Numbering

- **Location**: Throughout Section 5
- **Quote**: Section 5.2 references "Figure 3" but the caption describes dose-response curves; Section 5.4 H1 references "Figure 4" but the caption describes absorption vs. steering scatter; Section 5.4 H2 references "Figure 5" but the caption describes absorption vs. probing.
- **Problem**: The figure references in the text do NOT match the figure filenames in the `figures/` directory. The text says:
  - "Figure 2" = absorption rates (matches `fig2_absorption_rates.pdf`)
  - "Figure 3" = dose-response curves (but `fig3_absorption_vs_steering.pdf` is actually the scatter plot)
  - "Figure 4" = absorption vs. steering scatter (but `fig4_absorption_vs_probing.pdf` is the probing scatter)
  - "Figure 5" = absorption vs. probing (but `fig5_dose_response.pdf` is the dose-response)
  The text and filenames are cross-wired. In Section 5.2, the text references "Figure 3" for dose-response curves, but `fig3` is actually the steering scatter. In Section 5.4 H1, "Figure 4" is referenced for steering scatter, but `fig4` is the probing scatter.
- **Fix**: Reconcile figure numbering. Two options: (A) Change text references to match filenames: reference fig3 as "steering scatter" and fig5 as "dose-response", or (B) Rename figure files to match text. Option A is safer since figure files may be referenced elsewhere. The corrected mapping should be:
  - Figure 2: `fig2_absorption_rates.pdf` - absorption rates (correct)
  - Figure 3: `fig3_absorption_vs_steering.pdf` - absorption vs. steering scatter (text in 5.4 H1 should reference this)
  - Figure 4: `fig4_absorption_vs_probing.pdf` - absorption vs. probing scatter (text in 5.4 H2 should reference this)
  - Figure 5: `fig5_dose_response.pdf` - dose-response curves (text in 5.2 should reference this)

---

## Major Issues

### Issue 3: Missing Controls Results

- **Location**: Section 5.2 (Feature Steering Results)
- **Quote**: "Controls. We include a null steering condition (s = 0) to verify that the unmodified model produces stable baselines." (from Method section 4.4)
- **Problem**: The Method section (4.4) explicitly mentions a null steering control (s = 0) but the Experiments section never reports its results. The reader expects to see: (1) Did null steering produce stable baselines? (2) What were the baseline probabilities P_0(t)? (3) Were there any anomalies? This is important for validating the steering success metric.
- **Fix**: Add a brief paragraph or sentence in Section 5.2 reporting the null control results. Even a single sentence like "The null steering condition (s = 0) produced stable baselines with mean target-token probability P_0(t) = X.XX across all features, confirming that observed effects are due to steering rather than prompt variability" would suffice.

### Issue 4: Inconsistent Absorption Classification Thresholds

- **Location**: Section 5.1, Table 3; Method section 4.3
- **Quote**: Method defines "HIGH: A(f) > 0.10, MEDIUM: 0.05 <= A(f) <= 0.10, LOW: A(f) < 0.05". But Table 3 uses columns "HIGH (>=10%)" and "LOW (<10%)" with no MEDIUM column.
- **Problem**: The Method section defines three categories (HIGH/MEDIUM/LOW) but Table 3 collapses MEDIUM and LOW into a single "LOW (<10%)" column. This is inconsistent. The outline also planned three categories. Furthermore, the text says "18--26 of 26 features per layer show absorption below 10%" but the MEDIUM category (5-10%) should be separately tracked.
- **Fix**: Add a MEDIUM column to Table 3, or remove the MEDIUM category from the Method section if it was never used. If MEDIUM was not used in practice, update Method 4.3 to reflect the binary classification actually employed (HIGH >= 10%, LOW < 10%).

### Issue 5: Missing Layer 0 and Layer 10 Steering/Probing Results

- **Location**: Section 5.2, 5.3
- **Quote**: "We tested feature steering effectiveness at six strengths for layers 4 and 8." (Section 5.2)
- **Problem**: The Method section (4.2) states four layers were tested: 0, 4, 8, 10. Absorption detection results are reported for all four layers, but steering and probing are only reported for layers 4 and 8. The reader is left wondering: were layers 0 and 10 excluded from steering/probing? If so, why? The outline promised steering/probing across "multiple layers."
- **Fix**: Add a brief explanation for why only layers 4 and 8 have steering/probing data. If layers 0 and 10 were excluded due to resource constraints or because absorption variance was too low, state this explicitly.

### Issue 6: Probing F1 Range Claim Lacks Precision

- **Location**: Section 5.3
- **Quote**: "At k = 5, F1 scores ranged from 0.18 to 1.00 across features"
- **Problem**: The claim "0.18 to 1.00" is imprecise. Looking at the source data, the minimum F1 at k=5 is 0.182 (feature C at layer 4, feature S at layer 8 also 0.182). The maximum is 1.0 (feature X at both layers). But the text rounds 0.182 to 0.18, which loses meaningful precision. More importantly, the text does not specify *which* feature achieved the minimum or maximum, making it hard for readers to cross-reference.
- **Fix**: Report the exact values: "F1 scores ranged from 0.182 (feature C at layer 4) to 1.00 (feature X at both layers)." This gives readers anchor points.

### Issue 7: Power Analysis Mentioned but Not Leveraged

- **Location**: Section 5.1 (last paragraph), Method 4.6
- **Quote**: "This limited variance constrains the statistical power of our subsequent correlation analyses."
- **Problem**: The section mentions low variance as a power constraint but does not quantify what this means for interpreting the null results. The Method section (4.6) provides a power analysis (65% power for |r| >= 0.50, n=26) but this is never referenced in the Results when discussing the non-significant correlations. The reader needs to understand whether the null results are due to absence of effect or insufficient power.
- **Fix**: Add a sentence in Section 5.4 or 5.5 referencing the power analysis: "With n = 26 features and observed correlations in the -0.30 to +0.01 range, our study has limited power to detect small-to-medium effects. The 95% confidence interval for r = -0.301 (layer 8 H1) is approximately [-0.62, +0.10], which includes moderate negative correlations that would support H1."

---

## Minor Issues

- **Section 5.1, line 9**: "Layer 4 shows the highest mean absorption (0.039)" -> The value 0.039 is correct per source data, but the text should clarify whether this is a raw mean or weighted mean. The source data shows layer 4 mean = (0.077+0.146+0.133+0.148+0.160+0.140+0.099+0.113)/26 = 0.039, which is correct.

- **Section 5.2, line 26**: "HIGH-absorption features achieve mean success rates of 0.767 (layer 4) and 0.725 (layer 8)" -> These means are not directly verifiable from the source JSON without knowing which features are classified as HIGH. The text should either list the HIGH features per layer or move this to a table.

- **Section 5.2, line 28**: "while LOW-absorption features achieve 0.789 (layer 4) and 0.882 (layer 8)" -> Same issue as above - these aggregate statistics are not directly traceable to source data.

- **Section 5.3, line 34**: "Feature X achieves F1 = 1.00 at layer 4 with zero absorption; feature Z achieves F1 = 0.89 at layer 4, also with zero absorption." -> Feature Z F1 at layer 4 is 0.889 (from source), rounded to 0.89. Use exact value 0.889 for consistency with other numbers.

- **Section 5.4, Table 1**: The H3 row uses "--" for Layer, Pearson r, p-value, and R^2 columns, but these dashes create visual inconsistency. Consider using "N/A" or leaving blank.

- **Section 5.4, H3 paragraph**: "CV = sigma / |mu| = 0.932 exceeds the 0.5 threshold" -> The threshold of 0.5 was defined in Method 4.6 but not explained in the Results. A brief reminder like "exceeds our consistency threshold of CV < 0.5 (see Section 4.6)" would help.

- **Figure references**: The section uses "Figure 2", "Figure 3", etc. but the actual figure files use "fig2", "fig3", etc. This is a minor inconsistency in naming convention.

- **Table 2**: Feature S at layer 8 has F1 = 0.18 in the table, but source data shows 0.182. The table rounds to 2 decimal places while some F1 values in the text use 3. Standardize to 3 decimal places for consistency.

- **Section 5.1**: The text mentions "Figure 2 shows the absorption rate for each feature and layer" but the figure reference appears before the paragraph that introduces Table 3. Consider swapping the order: introduce Table 3 first, then reference Figure 2 for the visual distribution.

---

## Visual Element Assessment

- [x] Figures/tables match outline plan (partially - Table 3 is present but missing MEDIUM category; Figure 5/dose-response is present but figure numbering is cross-wired)
- [ ] All visuals referenced before appearance (NO - figure numbering is mismatched; Figure 3 caption in text describes dose-response but fig3 file is steering scatter)
- [x] Captions are self-explanatory (captions are descriptive and clear)
- [ ] No text-heavy sections that need visual support (NO - Section 5.3 on probing is entirely prose with no figure; the outline planned Figure 4 for absorption vs. probing but the text references it in 5.4, not 5.3)

### Missing Visual Elements

1. **No figure for probing results in Section 5.3**: The section is entirely text-based. The outline planned Figure 4 (absorption vs. probing scatter) but it is only referenced in 5.4 (Hypothesis Testing), not in 5.3 where the probing results are first presented. Consider adding a reference to Figure 4 in Section 5.3.

2. **No dose-response table**: Section 5.2 reports mean success rates by absorption category and strength, but these are buried in prose. A small table showing the dose-response matrix (strength x category) would improve clarity.

3. **Table 2 could include layer information more prominently**: The current Table 2 mixes layers 4 and 8 without clear visual separation.

---

## What Works Well

1. **Clear null result narrative**: The section effectively communicates the core finding. The structure (detection -> steering -> probing -> hypothesis testing) follows a logical progression that guides the reader through the evidence. Paragraph 3 of Section 5.2 ("Notably, even the most absorbed feature...") is an excellent concrete example that challenges intuition.

2. **Accurate data reporting**: The statistical values in Table 1 match the source JSON data exactly (r = +0.008 vs 0.0077 in JSON; r = -0.301 vs -0.3005 in JSON). This precision builds credibility. The Spearman correlations reported in the H1/H2 paragraphs also match the source data.

3. **Effective use of Table 2**: The top-absorbed-features table is well-constructed and makes the key point visually: high absorption does not preclude high steering success. Feature U (0.242 absorption, 1.00 steering success) is a compelling counterexample that supports the null result thesis.

4. **Good integration of statistical and narrative elements**: The hypothesis testing section (5.4) weaves together table data, figure references, and interpretation effectively. The "Not supported" classification is clear and consistent.
