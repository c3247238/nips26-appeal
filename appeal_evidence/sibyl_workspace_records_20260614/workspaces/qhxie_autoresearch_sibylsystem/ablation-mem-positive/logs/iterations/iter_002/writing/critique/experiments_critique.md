# Critique: 4. Experiments

## Summary Assessment

The Experiments section presents six hypothesis-driven experiments with thorough quantification and appropriate statistical testing. The structure is clear, each subsection follows a consistent Setup/Results format, and negative results (H3, H6) are reported honestly with revised interpretations. However, a critical discrepancy exists between the section's claim that CV analysis was conducted at $\lambda = 5 \times 10^{-5}$ and the JSON data showing it was actually conducted at $\lambda = 0.001$, creating a cross-section inconsistency that must be resolved. Additionally, several tables reference metrics that are not clearly defined in the section.

## Score: 7/10

**Justification**: The section is well-organized and methodologically sound, but the lambda mismatch between text and data, combined with a few missing method details and inconsistent non-absorbed feature statistics (n_non_absorbed = 0 but CV = 0.0 everywhere), weakens the credibility of the H4 "variance paradox" claims. Fixing the lambda discrepancy and clarifying the classification method would bring this to an 8.

## Critical Issues

### Issue 1: Lambda Mismatch in CV Analysis (H4)

- **Location**: Section 4.4, paragraph 1; Table 4; compare against `cv_full_analysis.json` config.lambda
- **Quote**: "We compute per-feature coefficient of variation $CV = \sigma / \mu$ across $n = 1000$ samples **at $\lambda = 0.001$**"
- **Problem**: The text explicitly states $\lambda = 0.001$, and the JSON data confirms $\lambda = 0.001$. This contradicts the outline's stated plan to measure CV "at $\lambda_c = 5 \times 10^{-5}$" and the H4 characterization as "at Critical Sparsity." The outline in `outline.md` line 123 says "at lambda=5e-5" but the experiment was run at lambda=0.001. This is a cross-section inconsistency between the Experiments section, the outline, and the proposal.
- **Fix**: Either (a) correct the text to state $\lambda = 0.001$ and update the outline/proposal to reflect that the CV analysis was run at standard sparsity rather than critical sparsity, or (b) rerun the CV analysis at $\lambda_c = 5 \times 10^{-5}$. Option (a) is the minimal fix and requires updating section 4.4 text, the outline's H4 description, and the proposal's H4 description.

### Issue 2: Non-Absorbed CV = 0.0 Throughout

- **Location**: Table 4 and `cv_full_analysis.json` layer results
- **Quote**: "CV (non-absorbed): 0.0" for all layers in Table 4
- **Problem**: n_non_absorbed = 0 for every layer (absorption_rate = 1.0), yet CV_non_absorbed = 0.0 is reported. This is arithmetically consistent (0/n = 0) but raises a credibility question: how can absorption_rate = 1.0 if there are non-absorbed features? The JSON data shows `n_non_absorbed: 0` and `absorption_rate: 1.0` for every layer. If ALL features are classified as absorbed, the CV comparison between absorbed and non-absorbed is meaningless. This needs explanation—either the thresholding method, the A_j > 0.001 criterion, or the sample statistics.
- **Fix**: Add a clarifying sentence: "At $\lambda = 0.001$, all SAE features exceed the absorption threshold ($A_j > 0.001$), resulting in n_non_absorbed = 0. The CV = 0.0 for non-absorbed reflects this boundary condition; genuine non-absorbed CV measurement requires lower sparsity." This frames the CV = 0.0 as a methodological boundary case, not a data artifact.

### Issue 3: Activation Patching Section Has No Method Description

- **Location**: Section 4.7, lines 108-124
- **Quote**: "Zeroing child features should recover parent activation if absorption is genuine." + Table 6
- **Problem**: No procedure is described for the activation patching experiment. How were the 9 persistent core words selected? What was the patching protocol (which layers, which features, what steering strength)? How was "recovery percentage" computed? The section presents results without methodological grounding.
- **Fix**: Add a 3-4 sentence experimental procedure: "We selected the 9 features (top features 22545, 3839, 4356, etc.) that were consistently identified across core words. For each feature, we computed the residual stream activation at layer 6 with the original SAE, then recomputed with the child feature zeroed out, and measured the recovery fraction."

## Major Issues

### Issue 4: Steering Effectiveness Section Has No Method Description

- **Location**: Section 4.8, lines 128-138; Table 7
- **Quote**: "We test whether CV predicts steering effectiveness by comparing high-CV versus low-CV absorbed features"
- **Problem**: No details on how features were selected as "high-CV" vs "low-CV," what steering strength values mean ($\tau \in \{\pm 3, \pm 5, \pm 7\}$), how the steering effect was measured (logit difference, which token), or the sample size (N = 30 per group).
- **Fix**: Add a brief procedure: "We selected 30 absorbed features with CV > 5 (high-CV group) and 30 absorbed features with CV < 1 (low-CV group). For each feature, we applied steering at $\tau \in \{\pm 3, \pm 5, \pm 7\}$ and measured the logit change at the target token. Mean steering effect was computed as the average $|\Delta_{logit}|$ across all steering strengths and samples."

### Issue 5: Co-occurrence Formula Not Explained

- **Location**: Section 4.5, line 82
- **Quote**: "We compare the baseline decoder cosine similarity against the revised co-occurrence score $S_{revised} = \cos(d_j, d_k) \cdot \log(f_j / f_k) \cdot (1 - \rho_j \rho_k)$"
- **Problem**: The formula appears without adequate explanation of its components. What are $\rho_j$ and $\rho_k$ specifically? The notation.md defines them as "normalized activation suppressions" but this is not self-contained. The text should be readable without constantly referencing notation.md.
- **Fix**: Add a brief explanatory clause: "where $\rho_j$ and $\rho_k$ are normalized activation suppressions derived from the encoder-decoder angle (Chanin et al., 2024)."

### Issue 6: Graph Topology Edges Not Defined

- **Location**: Section 4.6, line 92
- **Quote**: "edges represent parent-child absorption pairs ($A_j > 0.001$)"
- **Problem**: This is the definition of absorbed features, not edges. Edges in a graph represent relationships between pairs of features. The description should clarify: are edges drawn between feature j and feature k when A_j > threshold AND A_k > threshold? Or when the absorption is bidirectional? Or when one absorbs the other?
- **Fix**: Revise to: "Edges represent parent-child absorption pairs, where an edge connects feature j to feature k if feature j absorbs feature k (A_j > 0.001)."

### Issue 7: Table 4 Caption References "t-statistic" Without Definition

- **Location**: Table 4 header
- **Quote**: "t-statistic" column header in Table 4
- **Problem**: A t-statistic requires specifying which t-test and what the comparison groups are. The section mentions t-test in the context of H4 CV analysis, but Table 4 t-statistics are from comparing absorbed vs non-absorbed CV, which is different from what the text describes.
- **Fix**: Add table note: "t-statistic tests $H_0: \mu_{absorbed} = \mu_{non-absorbed}$ comparing CV distributions."

## Minor Issues

- **Line 16**: "1000 samples per point" — verify this matches the JSON (n_samples = 1000), confirmed consistent.
- **Line 40**: "to our knowledge, the first quantitative measurement" — This is a "first" claim that borders on hollow self-praise. Consider rephrasing to "representing, to our knowledge, the first quantitative measurement of this scaling law" or removing "to our knowledge."
- **Line 48-55 Table 3**: The table shows absorption_rate = 1.0 for all layers, but the JSON `absorption_stats` shows `n_absorbed: 4648` and `n_non_absorbed: 0`. If n_non_absorbed = 0, how can absorption_rate be 1.0 with N = 786432 (total features)? This needs clarification: the absorption_rate here is the fraction of tested candidate pairs that are absorbed, not the fraction of all SAE features.
- **Line 76**: "All differences are significant at $p < 0.01$" — The p-values from the JSON are exactly 0.0 for all layers (t-statistics exceed 1000). Reporting "p < 0.01" is technically true but loses the information that these are extreme significance levels. Consider stating "p < 10^-300" or noting the effective p-value is indistinguishable from zero given the large sample sizes.
- **Line 124**: "mean recovery $67.3\%$ (SD: $10.2\%$)": This is a single-sample mean with SD. The section should note this is across-word mean +/- std, not a statistical test result.

## Visual Element Assessment

- [ ] Figure 1 referenced before appearance (line 18: "Figure 1 shows the quasi-critical phase transition behavior"), but the figure is listed in the text comments without being embedded in the document.
- [x] All figures referenced before they appear in text (Figures 1-4 all referenced)
- [ ] Figure captions are placeholders (`.pdf` filenames used as alt text descriptions)
- [ ] No embedded figure files visible in the document — only markdown image links
- [ ] Tables are self-contained with clear headers
- [ ] No redundancy between figures and tables

## What Works Well

1. **Clear hypothesis-structure**: Each experiment is tied to a specific hypothesis (H1-H6), with explicit pass/fail criteria stated and results reported against those criteria. This makes the section easy to follow and verify.

2. **Honest negative result reporting**: H3 and H6 are reported as NOT_SUPPORTED with genuine interpretation, not suppressed or reframed as successes. This is methodologically sound and builds credibility.

3. **Statistical rigor in H1/H2**: The susceptibility peak, chi_ratio computation, and R^2 scaling collapse are properly quantified with pass criteria. The H2 result (nu=3, R^2=0.951) is compelling and reproducible.

4. **CV variance paradox framing**: Presenting the CV reversal as a "genuine discovery" (H4 SUPPORTED in reversed direction) rather than a failed hypothesis correctly handles the unexpected result and connects it to the actionability paradox.
