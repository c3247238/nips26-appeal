# Critique: Experiments

## Summary Assessment

The experiments section presents UAD results with commendable honesty about limitations. UAD achieves F1=0.725 with perfect recall on GPT-2 Small layer 8---a clear positive signal. However, critical methodological gaps remain: no random baseline, no ablations, no statistical testing, no false positive analysis, and DFDA's metric is fundamentally broken. The structural issues (missing E4, table numbering) compound these problems.

## Score: 5/10

**Justification**: The section gets the core numbers right and is admirably honest about DFDA's limitations. But the lack of baselines, ablations, and statistical testing is a serious methodological gap. To reach 7-8: add random baseline, ablations, and bootstrap CIs. To reach 9+: add false positive analysis, cross-model validation, and parent-positive DFDA evaluation.

---

## Critical Issues

### Issue 1: No Random Baseline
- **Location**: Sections 4.2, 5.1
- **Problem**: F1=0.725 is presented as a strong result, but without a random baseline, its magnitude is uninterpretable. There are C(500,2)=124,750 possible pairs among the top 500 features. UAD flags 51 pairs and finds 29 true positives. What would random selection achieve? If random selection of 51 pairs achieves F1~0.3, the contribution is marginal. If random achieves F1~0.05, the contribution is substantial.
- **Assessment**: This is the most critical missing experiment. It is computationally trivial (random sampling) and essential for interpretation.
- **Fix**: Compute random baseline: randomly select 51 pairs from top 500 features, compute precision/recall/F1 against Chanin labels. Repeat 100 times and report mean + std. This takes <1 minute.

### Issue 2: Multi-Seed "Robustness" Is Determinism
- **Location**: Section 4.3
- **Problem**: All three seeds produce IDENTICAL results---same 51 pairs, same 29 TPs, same supervised_labels dictionary. The JSON data confirms this: every field is identical across seeds. This is not robustness; it is complete determinism because the SAE weights are frozen and 1000 samples stabilize co-occurrence statistics.
- **Assessment**: Calling this "robustness" is misleading. The paper correctly notes this in the text but the section heading and abstract framing still suggest robustness.
- **Fix**: Rename the section. Do not claim robustness until tested across different SAE initializations or corpora.

### Issue 3: DFDA Metric Is Fundamentally Broken
- **Location**: Section 4.5, Table 4
- **Problem**: The 99.5% MSE improvement is artifactual. The MLP learns to predict near-zero parent values in child-dominant positions (where the parent is already suppressed). Baseline MSE measures deviation from zero; compensated MSE measures deviation from a near-zero prediction. The improvement is mathematically guaranteed because the MLP can always learn to output values close to zero.
- **Assessment**: This metric does not measure absorption recovery. It measures the MLP's ability to predict the mean of the training distribution (which is near-zero for child-dominant examples). The entire DFDA results section should be removed or heavily caveated.
- **Fix**: Either (a) remove DFDA entirely from the experiments section, or (b) relegate to a single paragraph with NO results table. Do not report the 99.5% figure.

### Issue 4: Cross-Layer Results Are Partial Failure, Not Validation
- **Location**: Section 4.4, Table 5
- **Problem**: Layer 4 achieves F1=0.432, below the 0.5 minimum threshold. The mean F1=0.561 is driven by layer 8's outlier performance. Without layer 8, the mean would be 0.490. Presenting "mean F1=0.561 across layers" as validation masks the layer-dependent failure.
- **Assessment**: UAD works well at layer 8 and poorly at layer 4. This is an important finding but should be presented honestly.
- **Fix**: Report layer-wise results individually. State clearly: "UAD performs well at layer 8 (F1=0.725) but falls below threshold at layer 4 (F1=0.432), suggesting absorption signatures are strongest in mid-to-late layers."

---

## Major Issues

### Issue 5: No Ablation Studies
- **Location**: Methodology, Experiments
- **Problem**: The paper claims UAD uses HAC with Ward linkage, phi coefficient normalization, and top-500 feature selection, but never tests whether these choices matter. Would k-means work? Would raw co-occurrence achieve similar F1? Would spectral clustering (as in "Geometry of Concepts") outperform HAC?
- **Assessment**: Without ablations, the method's design choices are unjustified. A reviewer will ask: why HAC? Why phi coefficient? Why 50 clusters?
- **Fix**: Run ablations: (1) k-means vs HAC, (2) raw co-occurrence vs phi coefficient, (3) 25 vs 50 vs 100 clusters, (4) top 250 vs 500 vs 1000 features. Each takes ~10 seconds.

### Issue 6: No Statistical Significance Testing
- **Location**: All results sections
- **Problem**: No p-values, confidence intervals, or bootstrap estimates are reported. The notation table defines bootstrap CI but notes it is "not yet implemented." Without statistical testing, the F1=0.725 claim has no grounding.
- **Assessment**: A reviewer will rightly ask: is F1=0.725 significantly different from random? Is the cross-layer difference significant?
- **Fix**: Implement bootstrap CI for F1 (resample examples with replacement 1000 times). Compute permutation test p-value (shuffle cluster assignments under null).

### Issue 7: No False Positive Analysis
- **Location**: Section 4.2
- **Problem**: With precision=0.569, 22 of 51 same-cluster pairs are false positives. The paper acknowledges this but never analyzes what these false positives are. Understanding false positives would strengthen the method and guide post-hoc filtering.
- **Assessment**: This is a missed opportunity. Analyzing false positives could suggest filters to raise precision above 0.8.
- **Fix**: Analyze the 22 false positive pairs: report their phi coefficients, marginal frequencies, and semantic patterns. Categorize them (topical co-occurrence, shared super-absorber, random).

---

## Minor Issues

- **Table 4 phi values**: Identical phi (0.812) for pairs 1-4 and 6 is suspicious. Verify or explain.
- **Missing E4**: Experiment numbering jumps from E3 to E5.
- **"15,000 token positions"**: Clarify calculation and justify sufficiency.
- **PARTIAL_PASS terminology**: Should not appear in paper draft.
- **Figure references**: Figures 2, 3, 4 referenced but not verifiable.

---

## What Works Well

1. **Honest DFDA caveat**: The explicit disclosure that the metric is artifactual builds reviewer trust.
2. **Determinism distinction**: Correctly identifies that perfect consistency reflects determinism, not robustness.
3. **Precision contextualization**: The "screening tool, not classifier" framing is honest and appropriate.
4. **Pilot-to-full structure**: The narrative flow from pilot to full experiments is logical and clear.
5. **Table 5 comprehensive summary**: Well-structured overview of all conditions.
