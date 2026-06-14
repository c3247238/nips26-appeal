# Critique: Experiments

## Summary Assessment
The Experiments section presents three clearly structured experiments with specific numbers and statistical tests. However, it contains a critical inconsistency between reported numbers and source data, omits important methodological caveats from the pilot, and makes a causal claim that overreaches the regression design. The section reads well but would not survive a rigorous reviewer at a top venue without revision.

## Score: 5/10
**Justification**: The structure and clarity are above average, but a critical data mismatch (E1 numbers), an unsupported causal claim, and missing limitations drag the score down. Fixing the data alignment and softening the causal language would bring it to 7/8.

## Critical Issues

### Issue 1: Experiment 1 numbers do not match the source data
- **Location**: Table 1 and paragraph 2 of Section 4.1
- **Quote**: "Feature-splitting checkpoints achieve 0% dead neurons and strong explained variance (0.976–0.986), but do not dominate on hedging or absorption. Table 1 reports per-family averages." / Table 1 shows Standard (n=23) L0=33.9, EV=0.830, dead=0.197, CE Rec=1.054, absorption=0.015, hedging=0.833.
- **Problem**: The source log (`e1_full_gpt2.log`) and summary (`e1_full_gpt2_summary.md`) show that the **only** checkpoint with non-zero absorption in the full E1 run is `hook-z-kk-l8` (alpha = 0.345). All 23 other standard checkpoints and all 4 feature-splitting checkpoints have absorption = 0.0. Therefore the Standard family average of 0.015 is technically correct (0.345/23 = 0.015), but it is misleading because it implies variance that does not exist—22 of 23 standard checkpoints are exactly 0.0. More seriously, the text says "Standard checkpoints occupy a broad region... The hook-z SAE is the only standard checkpoint with non-zero absorption"—this is correct. But Table 1 presents the family average without noting that the standard deviation is driven by a single outlier, which is poor statistical practice for a 4/7 venue.
- **Fix**: Add a footnote to Table 1 clarifying that the Standard absorption mean is driven entirely by one checkpoint (`hook-z-kk-l8`, alpha=0.345) and that 22/23 Standard checkpoints show zero absorption. Better yet, report the median or add a column for standard deviation.

### Issue 2: Causal language overreaches the regression design
- **Location**: Section 4.2, final paragraph: "These results support H2: absorption carries a unique negative causal cost for downstream interpretability utility, independent of confounders."
- **Quote**: "absorption carries a unique negative causal cost for downstream interpretability utility, independent of confounders"
- **Problem**: This is an observational regression on pretrained checkpoints, not a randomized experiment. The Discussion (Section 7.3) correctly notes: "We report associations conditional on these controls, not definitive causal effects." The Experiments section contradicts this by using the word "causal" twice ("causal cost" and "unique negative causal effect").
- **Fix**: Replace "causal cost" with "predictive cost" or "unique negative association." If you want to keep "causal," add a qualifier: "consistent with a causal interpretation" or "suggestive of a causal link."

### Issue 3: Missing caveat about simplified absorption metric in E1 and E3
- **Location**: Section 4.1 and 4.3
- **Quote**: "Metrics include absorption rate (alpha), hedging rate (h)..." / "We construct a task-agnostic absorption metric... then compare it to the simplified first-letter benchmark"
- **Problem**: The `pilot_summary.md` explicitly states: "The simplified first-letter proxy is too coarse and does not align with the rigorous sae-spelling benchmark. A proper absorption evaluation requires the full Chanin et al. pipeline." This limitation is never mentioned in the Experiments section. A reviewer will ask why the paper uses a "simplified" proxy without acknowledging its crudeness.
- **Fix**: Add a sentence in Section 4.1 and 4.3 noting that the first-letter absorption metric is a simplified proxy and that the degeneracy (0.0 on most checkpoints) may reflect the proxy's coarseness rather than true zero absorption. This is essential for scientific honesty and aligns with the Discussion's limitations.

## Major Issues

### Issue 4: Table 2 regression coefficients differ slightly from the source data
- **Location**: Table 2 in Section 4.2
- **Quote**: Table 2 reports absorption coefficients as -0.037***, -0.022***, -0.023*** for Sparse Probing F1, RAVEL Cause, and RAVEL Isolation respectively.
- **Problem**: The source `e2_meta_summary.md` reports the coefficients as -0.0373, -0.0216, and -0.0235. The rounding in Table 2 is acceptable, but the RAVEL Cause coefficient rounds from -0.0216 to -0.022, which changes the second decimal place. More importantly, the standard errors in Table 2 are listed as (0.006) for all three, but the source data shows SEs of 0.0055, 0.0057, and 0.0057. Rounding all to 0.006 is imprecise and makes the coefficients look more uniform than they are.
- **Fix**: Report coefficients and SEs to three decimal places (matching the source data): -0.037 (0.006), -0.022 (0.006), -0.024 (0.006). Or better, use the exact values from the summary: -0.037 (0.006), -0.022 (0.006), -0.024 (0.006).

### Issue 5: Figure references are inconsistent with the outline plan
- **Location**: Section 4.2, paragraph 3
- **Quote**: "Figure 2 plots the partial-regression relationship between absorption and sparse probing F1."
- **Problem**: The outline plan (outline.md) specifies Figure 2 as "Partial Correlation Scatter — Absorption vs. Sparse Probing F1" and Figure 4 as "Combined bar chart of standardized absorption regression coefficients and partial-correlation scatter." The Experiments section never mentions Figure 4, but the Discussion section (Section 7.1) references Figure 4. This means Figure 2 and Figure 4 may be redundant or misnumbered.
- **Fix**: Clarify the figure numbering. If Figure 2 is the partial-correlation scatter in the Experiments section, then the Discussion's Figure 4 should either be removed or renumbered to Figure 2 (with a cross-reference). Ensure each figure appears exactly once and is numbered sequentially.

### Issue 6: Missing baseline comparison in E1
- **Location**: Section 4.1
- **Quote**: "17 of 27 checkpoints lie on the empirical Pareto front."
- **Problem**: The section reports how many checkpoints are Pareto-optimal but does not break this down by family in a way that supports the "no dominance" claim. The reader cannot see, from the text alone, whether feature-splitting or standard checkpoints are more represented on the front.
- **Fix**: Add a sentence breaking down Pareto representation by family: e.g., "All 4 feature-splitting checkpoints and 13 of 23 standard checkpoints lie on the Pareto front." This is in the source summary and should be in the text.

### Issue 7: E3 sample size and significance are underplayed
- **Location**: Section 4.3
- **Quote**: "Pearson r = -0.592, Spearman rho = -0.529, p = 0.116. The correlation is negative and not statistically significant."
- **Problem**: While the p-value is reported, the section does not explicitly state that N=10, which is a major limitation. The negative correlation is driven almost entirely by a single outlier (TopK_Attn, alpha_FL=0.654, alpha_TA=0.000). Without this outlier, the correlation would likely be near zero.
- **Fix**: Add "(N=10)" after the first sentence of the Results paragraph. Explicitly note the outlier's leverage: "Removing the single TopK_Attn outlier reduces the correlation to approximately zero." This is in the source summary and should be highlighted.

## Minor Issues

- **Section 4.1, line 7**: "CE loss recovered" is abbreviated as "CE Rec." in Table 1 but spelled out elsewhere. Be consistent with the glossary, which uses "CE loss recovered."
- **Section 4.2, line 32**: "OLS with cluster-robust standard errors confirms absorption as a significant negative predictor" — change "confirms" to "shows" or "indicates" to avoid overstating.
- **Section 4.3, line 57**: "The first-letter metric is degenerate on 9 of 10 checkpoints" — the glossary defines "degenerate" in the context of metrics, but this term may confuse readers. Consider "collapses to zero" or "is uninformative."
- **Table 3**: The "Family" column uses "TopK_MLP" and "TopK_Attn" but the glossary and Method section use "TopK_MLP" and "TopK_Attn" inconsistently (Method says "TopK_MLP (MLP-output hook)" and "TopK_Attn (attention-output hook)"). Ensure capitalization is consistent.
- **Missing cross-reference**: The Method section (Section 3.2) says the task-agnostic metric uses "cosine similarity > tau (we set tau = 0.7)". The Experiments section never mentions this threshold. Add a brief note in E3 for reproducibility.

## Visual Element Assessment
- [x] Figures/tables match outline plan (mostly, but Figure 2/4 numbering needs clarification)
- [x] All visuals referenced before appearance
- [ ] Captions are self-explanatory: Figure 1 caption mentions "stars mark Pareto-optimal points" but does not define what the axes represent (absorption rate vs. hedging rate) in plain language.
- [ ] No text-heavy sections that need visual support: Section 4.2 describes regression results in text and Table 2, but a small visualization of the standardized coefficients (as planned in Figure 4) would strengthen the section.

## What Works Well
1. **Clear hypothesis-test mapping**: Each subsection ends with an explicit statement of whether H1/H2/H3 is supported. This makes the section easy to follow and aligns perfectly with the outline.
2. **Specific statistical reporting**: The section reports exact test statistics (U = 48.0, p = 0.754; r_partial = -0.385, p < 0.001) rather than vague claims of significance. This is reviewer-friendly.
3. **Honest negative result in E3**: The section does not try to spin the non-significant correlation into a positive finding. Instead, it frames the negative result as evidence that the first-letter benchmark may be unrepresentative. This is intellectually honest and strengthens the paper's credibility.
