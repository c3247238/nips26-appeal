# Critique: Limitations and Future Work

## Summary Assessment
This section provides a candid enumeration of limitations that strengthens the paper's credibility. The seven limitations are specific and well-articulated, and the future work directions map sensibly to each limitation. However, the section lacks integration with the paper's core theoretical claims---it reads as a generic checklist rather than a thoughtful reflection on where the inhibition framework itself might break down. The ordering is also suboptimal, burying the most serious methodological concern (point 6: single significant result) in the middle of the list.

## Score: 6/10
**Justification**: The section is competent but uninspired. It lists limitations without prioritizing them, misses opportunities to connect limitations to the theoretical framework, and contains one claim that contradicts the experiments section. To reach 7-8, it needs: (1) clear prioritization of limitations by severity, (2) explicit discussion of where the competitive suppression theory itself might fail, and (3) removal of the unsupported claim about "low absorption variance limiting correlation power."

## Critical Issues

### Issue 1: Missing Discussion of Where the Inhibition Theory Itself Could Fail
- **Location**: Entire section (7.1)
- **Quote**: "[All seven limitations are empirical/scoped, none are theoretical]"
- **Problem**: The section never asks "what if competitive suppression is wrong?" Every limitation is about sample size, model scale, or measurement breadth---none address whether the core theoretical claim (decoder correlations = competitive suppression = absorption mechanism) could be incorrect. A top venue reviewer will ask: what evidence would falsify the LCA correspondence? The section should address this.
- **Fix**: Add a limitation about the structural correspondence being approximate for untied weights (the standard case). The proof in Section 3.1 assumes tied weights, yet all experiments use untied SAEs. The section should explicitly flag this gap and state what would falsify the approximate correspondence (e.g., if precision@20 on untied SAEs is at chance).

### Issue 2: Unsupported Claim About "Low Absorption Variance"
- **Location**: Point 7, line 11
- **Quote**: "Low absorption variance. Most features show near-zero absorption, limiting correlation power and the generalizability of our findings to feature sets with stronger absorption."
- **Problem**: This claim is circular and misleading. The low variance is a *finding* (absorption is sparse), not a *limitation* of the study design. The experiments section (Section 5.4) explicitly states that "the H6--H10 experiments address this limitation by changing the prediction target"---so this limitation is already resolved by the paper's own design. Presenting it as an unresolved limitation contradicts the experiments section and weakens the paper.
- **Fix**: Remove this point or reframe it: "The first-letter feature set exhibits limited absorption variance, which constrained the H1--H3 correlation analyses. The H6--H10 framework-based tests are designed to be robust to this constraint."

### Issue 3: Missing Multiple Comparisons Correction Discussion
- **Location**: Entire section
- **Quote**: "[No mention of multiple comparisons]"
- **Problem**: The experiments section (Table 1, Section 5.1) notes that "no hypothesis survives multiple comparison correction" with 12 tests. The limitations section completely omits this. For a paper with only one uncorrected significant result (H1b at layer 8, p=0.028), the lack of multiple comparisons correction is a serious limitation that should be front-and-center.
- **Fix**: Add as a major limitation: "Multiple comparisons. With 12 tests across H1--H3 and two layers, the single significant result (H1b, p=0.028) does not survive Bonferroni or BH-FDR correction. The H6--H10 experiments use structural predictions with chance baselines, reducing dependence on p-value thresholds, but the prior empirical foundation rests on uncorrected tests."

## Major Issues

### Issue 4: Limitations Are Not Prioritized by Severity
- **Location**: Section 7.1, ordering of points 1--7
- **Problem**: The limitations are listed in what appears to be arbitrary order. Point 6 (single significant result / family-wise error) is the most serious methodological concern but appears sixth. Point 1 (single model family) is important but less severe than the statistical power issue. A reviewer will scan the first 2--3 limitations and form an impression; burying the critical ones wastes credibility.
- **Fix**: Reorder by severity: (1) Single significant result / multiple comparisons, (2) Single model family, (3) Narrow feature set, (4) Small model, (5) Single absorption metric, (6) Two downstream tasks, (7) Low absorption variance (or remove).

### Issue 5: Future Work Lacks Connection to Limitations
- **Location**: Section 7.2
- **Problem**: The future work items (1--7) are sensible but read as a generic research agenda. Each item should explicitly map back to the limitation it addresses. For example, item 1 ("Test with authenticated Gemma/Pythia access") clearly addresses limitation 1 (single model family), but this connection is never stated.
- **Fix**: Restructure 7.2 as a mapping: "To address [Limitation X], we propose [Future Work Y]." Or add a column to the list showing the mapping.

### Issue 6: Missing Limitation on H6--H10 Experiment Execution Status
- **Location**: Entire section
- **Problem**: The experiments section (Section 5.2) describes H6--H10 as validation experiments with predictions, but the results for H6--H10 are not reported in the experiments section (only H1--H5 results appear). The limitations section should acknowledge if H6--H10 results are pending or preliminary.
- **Fix**: If H6--H10 results are indeed pending, add: "The validation experiments for H6--H10 are planned but not yet executed. The predictions in Section 5.2 are pre-registered; reported results in this paper are limited to H1--H5." If they have been executed, the experiments section needs updating.

### Issue 7: Inconsistent Framing of "Narrow Feature Set"
- **Location**: Point 2, line 2
- **Quote**: "Semantic features (e.g., WordNet hierarchies) may exhibit stronger absorption and clearer task degradation."
- **Problem**: This phrasing subtly suggests that first-letter features are a poor proxy for "real" features, which undermines the paper's empirical foundation. The intro and experiments sections treat first-letter features as a valid test case. The limitation should be framed neutrally.
- **Fix**: "First-letter features (A--Z) have a shallow, uniform hierarchy. Semantic features (e.g., WordNet hierarchies) may exhibit different absorption patterns, and the inhibition framework's predictions should be validated on feature sets with richer hierarchical structure."

## Minor Issues

- **Line 1 ("Single model family.")**: The parenthetical "(124M parameters)" is redundant with the intro. Remove to keep the limitation statement crisp.
- **Line 4 ("Single absorption metric.")**: "SAEBench's ablation-based metric or alternative measures may yield different results" --- vague. Specify what "different results" means: different absorption pairs identified, different correlation magnitudes, or different feature rankings?
- **Line 5 ("Two downstream tasks.")**: "Circuit finding and model editing, which require precise feature isolation, may be more sensitive to absorption" --- "may" is weak. Replace with "are expected to be" or provide a brief justification.
- **Line 6 ("Single significant result.")**: The phrase "this result could arise by chance" understates the issue. With 12 tests at alpha=0.05, the expected number of false positives is 0.6; observing 1 is not surprising. Be more direct: "With 12 uncorrected tests, a single p=0.028 result has a high probability of being a false positive."
- **Section 7.2, item 7**: "Investigate why the delta steering effect is layer-dependent" --- this is framed as a future work item but is already discussed in the discussion section (Section 6.2: "Layer-dependent effects reflect depth-varying inhibition strength"). Either remove this item or reframe it as an empirical validation of the theoretical prediction.
- **Missing transition**: The section lacks a transition sentence from the discussion. Add: "While the inhibition framework provides a unified explanation for our empirical findings, several limitations constrain the scope and confidence of our conclusions."

## Visual Element Assessment
- [x] Figures/tables match outline plan (no visuals planned for this section)
- [x] All visuals referenced before appearance (N/A)
- [x] Captions are self-explanatory (N/A)
- [ ] No text-heavy sections that need visual support --- The seven limitations would benefit from a summary table mapping each limitation to its severity (Critical/Major/Minor), the section it affects, and the future work item that addresses it.

## What Works Well
1. **Specificity of limitations.** Each limitation is concrete and measurable ("GPT-2 Small, 124M parameters", "26 first-letter features", "two downstream tasks"). This is far better than generic "limited sample size" complaints.
2. **Honesty about the single significant result.** Point 6 directly addresses the family-wise error rate concern, which builds reviewer trust. Most papers would bury this.
3. **Future work directions are actionable.** Items like "Test with authenticated Gemma/Pythia access" and "Try alternative absorption metrics" are specific enough that a reader could execute them. This is a sign of genuine scientific thinking rather than hand-waving.
