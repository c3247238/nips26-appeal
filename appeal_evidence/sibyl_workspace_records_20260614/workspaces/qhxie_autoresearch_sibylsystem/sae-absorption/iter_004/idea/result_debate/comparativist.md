# Comparativist Analysis: Competitive Geometry of Feature Absorption in SAEs

**Agent**: sibyl-comparativist
**Date**: 2026-04-14
**Iteration**: 004 (current)
**Model used**: GPT-2 Small (open-model anchor; Gemma-2-2b originally planned but gated)

---

## 1. Baseline Landscape

The table below summarizes the top existing methods for measuring, predicting, and mitigating feature absorption in SAEs. These are the primary comparators for the five contributions claimed in the proposal.

| Method | Source | Year | Key Result | Benchmark |
|--------|--------|------|------------|-----------|
| Chanin et al. absorption metric | arXiv:2409.14507 (NeurIPS 2025) | 2024 | Absorption rate 15-35% across Gemma Scope SAEs; probe-directed, requires known features | First-letter task, Gemma-2-2b |
| SAEBench absorption task | arXiv:2503.09532 (ICML 2025) | 2025 | Standardized absorption + 7 other metrics across 200+ SAEs; no downstream correlation analysis reported | Gemma-2-2b, 8 architectures |
| OrtSAE | arXiv:2509.22033 | 2025 | 65% absorption reduction via orthogonality penalty | Gemma-2-2b layer 12 |
| ATM SAE | arXiv:2510.08855 | 2025 | Mean absorption 0.0068 vs TopK 0.1402 and JumpReLU 0.0114 | Gemma-2-2b |
| Matryoshka SAE | arXiv:2503.17547 (ICML 2025) | 2025 | Best on SAEBench absorption, RAVEL, sparse probing | Gemma-2-2b |
| Masked Regularization | arXiv:2604.06495 | 2026 | Disrupts co-occurrence patterns; reduces absorption across architectures | Gemma-2-2b |
| Feature Sensitivity | arXiv:2509.23717 | 2025 | Frames absorption as special case of low feature sensitivity | Multiple models |
| DeepMind negative results | Medium blog, 2025 | 2025 | Dense probes dramatically outperform SAE probes on safety tasks; SAE research deprioritized | Chat data, safety tasks |

---

## 2. Contribution Margin Assessment

### Contribution 1: LV Competition Coefficient as Unsupervised Absorption Detector (H1)

**Claimed**: alpha_ij = sigma_ij * (f_j/f_i) predicts absorption with F1 > 0.65.

**Actual result**: Test F1 = 0.128 at best tau (0.5). ROC-AUC = 0.148. The LV detector **performs worse than the cosine-similarity-only baseline** (cosine test F1 = 0.165, ROC-AUC = 0.201). F1 delta vs. cosine = -0.037.

**Sharpness test**: The sigmoid vs. linear AIC comparison yields a winner of "linear" (AIC_sigmoid = -60.95, AIC_linear = -61.05). The LV theory predicts a step-function transition at alpha ~ 1; the data shows no such transition. The bin_rates show **highest absorption in the lowest alpha bin** (0.05: rate 0.848), which is the opposite of the predicted direction.

**Cross-architecture generalization**: On v5-32k, fixed-tau F1 = 0.009; on v5-128k, F1 = 0.000. Complete failure.

**Verdict**: H1 is **decisively falsified**. The LV competition coefficient does not predict absorption. The contribution margin is **negative** -- this method underperforms the trivial cosine baseline. This is a null result.

**Comparison to SOTA**: No prior work has proposed a Lotka-Volterra unsupervised detector, so this is novel -- but it is a novel failure. The LessWrong "Looking for Feature Absorption Automatically" post similarly reported negative results for cosine-similarity-based automatic detection, making this part of a growing body of evidence that unsupervised absorption detection from activation statistics alone is very difficult.

### Contribution 2: Corpus PMI Prediction of Absorption (H2)

**Claimed**: PMI predicts absorption with Pearson r > 0.50 after controlling for SAE config.

**Actual result**: PMI coefficient is **negative** (-0.006, wrong sign), non-significant (p = 0.593), with partial R^2 = 0.0006 (criterion was 0.10). Per-letter Pearson r = -0.080 (p = 0.699). Sign is inconsistent across layers.

**Verdict**: H2 is **clearly not supported**. Corpus co-occurrence statistics do not predict which feature pairs are absorbed. The contribution margin is **zero** -- a clean null result.

**Comparison to SOTA**: Masked regularization (arXiv:2604.06495) disrupts co-occurrence patterns and reduces absorption, which was taken as indirect evidence that co-occurrence drives absorption. Our direct test of this hypothesis produces a null result. This is informative: it suggests that while disrupting co-occurrence during training helps (an interventional result), co-occurrence statistics do not predict absorption beforehand (an observational failure). This distinction is worth reporting but is a weaker contribution than the predicted positive finding.

### Contribution 3: Downstream Impact Analysis (H3)

**Claimed**: Absorption-downstream correlation |r| < 0.2 (disconnection hypothesis).

**Actual result**: H3 is **falsified in the strong positive direction**. Across 54 Gemma-2-2b SAEs from SAEBench:
- Sparse probing F1: Pearson r = -0.595 (p < 0.001)
- SCR: Pearson r = -0.431 (p = 0.002)
- RAVEL proxy (TPP): Pearson r = -0.454 (p = 0.0006)
- Unlearning: r = -0.175 (not significant)

Partial correlations controlling for width, layer, and architecture are even stronger (sparse probing: partial r = -0.661, SCR: partial r = -0.677).

**Matched RAVEL comparison** (5 low vs 5 high absorption SAEs): Low-absorption mean TPP = 0.046 vs high-absorption mean = 0.009. Paired t-test: t = 4.27, p = 0.006, Cohen's d = 2.13 (large). SCR difference: t = 7.19, p < 0.001.

**Safety probe pilot**: Dense probe AUC 0.998-1.000; 1-sparse AUC 0.852-0.947. Probe gap Pearson r with absorption = -0.759 (n=3, p = 0.45 -- underpowered but directionally interesting).

**Verdict**: This is the **strongest finding** of the paper. H3's falsification demonstrates that absorption genuinely degrades downstream SAE quality. The effect sizes are substantial and survive Bonferroni correction and covariate control.

**Comparison to SOTA**: SAEBench (Karvonen et al., 2025) provides the raw data and observes that "proxy metrics do not reliably predict practical performance" -- but they do NOT report the systematic absorption-downstream correlation analysis that this work performs. DeepMind's 2025 blog post reports qualitative negative results for SAEs on safety tasks but does not quantify the absorption-performance link. **This is the first systematic quantification of the absorption-downstream relationship with proper statistical controls**, and the result is strong.

**However, a critical confound**: The high-absorption SAEs are overwhelmingly 1M-width SAEs, while low-absorption SAEs are 16k-65k. Width is a massive confound. The partial correlations attempt to control for this but with only 54 data points and highly correlated predictors (width and absorption), the partial correlation estimates may be unreliable. A reviewer will immediately point this out.

### Contribution 4: Width Paradox / DAS(k=3) (H4)

**Claimed**: DAS(k=3) increases monotonically with width while DAS(k=1) is non-monotone.

**Actual result**: PARTIAL. DAS(k=1) is indeed roughly non-monotone (mean: 24k=0.105, 49k=0.104, 98k=0.119). But DAS(k=3) does NOT increase monotonically: mean values are 24k=0.320, 49k=0.227, 98k=0.260. Only 42.3% of letters show positive k=3 slope (criterion was 80%).

**Verdict**: H4 is **not supported**. The width paradox analysis does not produce the predicted monotonic increase in distributed absorption. The k=1 non-monotonicity finding is mildly interesting but not surprising given known results.

**Comparison to SOTA**: No prior work measures DAS(k=3) or proposes the distributed absorption concept. The concept is novel but the empirical validation fails.

### Contribution 5: Absorption Taxonomy (Type I/II/III)

**Result**: Type I (full): 1/26 letters (3.8%). Type II (partial): 23/26 (88.5%). Type III (distributed): 0/26. Comprehensive rate: 92.3%.

**Critical problem identified by the experiment itself**: The Type II rate of 88.5% is flagged as "CRITICAL: likely inflated" in the evidence quality section. The parent features were identified by selectivity heuristics rather than sae-spelling ground truth. The magnitude ratio comparison uses a flawed fallback (global mean-when-active vs letter-token activation), making the Type II threshold systematically trigger. The causal link between the measured magnitude deficit and actual absorption is not established.

**Verdict**: The taxonomy concept is **interesting but the implementation is unsound**. The 92.3% comprehensive rate cannot be trusted. This contribution needs fundamental redesign before it is publishable.

---

## 3. Concurrent Work Scan

### Direct competitors (last 6 months, to April 2026):

1. **Masked Regularization** (arXiv:2604.06495, April 2026): Directly addresses absorption via training-time co-occurrence disruption. Our PMI null result is complementary but weaker.

2. **Sanity Checks for SAEs** (arXiv:2602.14111, Feb 2026): Shows SAEs recover only 9% of true features in synthetic settings and random baselines match SAEs. This is deeply concerning context for any absorption-focused paper.

3. **HSAE / Feature Forest** (arXiv:2602.11881, Feb 2026): Jointly learns SAE hierarchy with structural constraints. Could subsume the absorption taxonomy contribution.

4. **SynthSAEBench** (arXiv:2602.14687, Feb 2026): Provides synthetic ground-truth benchmarks for studying absorption. Could provide the controlled setting needed for the LV detector that real data failed to deliver.

5. **DeepMind SAE deprioritization** (March 2025): The blog post explicitly cites feature absorption as a key reason dense probes outperform SAEs. Our C3A result (absorption correlates with downstream degradation) provides the quantitative evidence for this claim.

### No directly competing paper found for:
- Lotka-Volterra / competitive exclusion framing for absorption (novel but failed)
- Systematic absorption-downstream correlation with Bonferroni correction on SAEBench data (novel and successful)
- Three-type absorption taxonomy (novel but methodologically flawed)

---

## 4. Novelty Verdict

**The ONE thing this work does that no prior work does**: It provides the first systematic, statistically controlled quantification that SAE absorption score is meaningfully correlated with downstream task degradation (r = -0.43 to -0.60 across three SAEBench tasks, surviving Bonferroni correction and covariate control).

This single finding is the paper's real contribution. Everything else either failed (LV detector, PMI prediction, width paradox) or is methodologically suspect (taxonomy).

---

## 5. Venue Recommendation

**Recommended venue**: Workshop paper at NeurIPS 2026 or ICLR 2027, or **mid-tier venue** (AAAI, EMNLP workshop track).

**NOT recommended for top-tier (NeurIPS/ICML main conference)** for the following reasons:

1. **Three of four hypotheses failed** (H1, H2, H4). The LV detector -- the paper's theoretical centerpiece -- decisively fails.
2. **Model mismatch**: All experiments use GPT-2 Small instead of the planned Gemma-2-2b. The original target model was gated. This limits the impact and generalizability. The SAEBench correlation analysis (C3A) does use Gemma-2-2b data, but from pre-computed scores, not from our own experiments.
3. **Width confound in C3A**: The absorption-downstream correlation is likely confounded by SAE width (1M SAEs dominate the high-absorption group). While partial correlations try to control for this, the confound is severe.
4. **Taxonomy is unsound**: The Type II classification is acknowledged as inflated.

**What would elevate to top-tier**: Replicate C3A on a model where width can be properly controlled (e.g., matched-L0, matched-width SAEs), successfully validate the LV detector on Gemma-2-2b (the model where Chanin et al. validated absorption), and fix the taxonomy methodology.

**Comparable papers at workshops**: The LessWrong "Looking for Feature Absorption Automatically" (negative result, community discussion); the Chanin et al. hedging paper (workshop-level contribution).

---

## 6. Strengthening Plan

### Critical additions (would maximally strengthen positioning):

1. **Replicate C3A on Gemma-2-2b with proper width controls**: Download Gemma-2-2b SAEBench data, and specifically compare SAEs of the SAME width and different L0 settings. If absorption still correlates with downstream degradation within width-matched pairs, the confound objection is eliminated. This is the single most important additional analysis.

2. **Run LV detector on Gemma-2-2b**: The original proposal targeted Gemma-2-2b for good reason -- Chanin et al. validated absorption there. GPT-2 Small has very different SAE training dynamics. The LV detector may perform differently on the original target. If it still fails, the negative result is more convincing.

3. **Add ATM SAE and Matryoshka SAE as baselines in C3A**: Currently, C3A only compares Gemma Scope JumpReLU SAEs. Adding ATM (absorption 0.0068) and Matryoshka SAEs to the correlation analysis would dramatically strengthen the range of absorption values and architectural diversity.

### Desirable but not critical:

4. **Causal intervention**: Run activation patching to verify that the absorption-downstream correlation is causal (ablating absorbed features hurts downstream performance). This would be the definitive test.

5. **Fix taxonomy with sae-spelling ground-truth parent features**: Use Chanin et al.'s actual parent feature identification (via LR probes) rather than the selectivity heuristic. This would make the taxonomy trustworthy.

---

## 7. Summary Assessment

| Contribution | Status | Contribution Margin | Novelty |
|-------------|--------|-------------------|---------|
| LV unsupervised detector (H1) | FAILED | Negative (worse than cosine baseline) | Novel framing, novel failure |
| Corpus PMI predictor (H2) | NULL | Zero (PMI non-significant, wrong sign) | Novel null result |
| Downstream impact (H3) | STRONG | Strong (r = -0.43 to -0.60) | Novel quantification |
| Width paradox / DAS(k=3) (H4) | PARTIAL | Marginal | Novel metric, inconclusive result |
| Absorption taxonomy | UNSOUND | Not assessable (inflated Type II) | Novel concept, flawed execution |

**Bottom line**: This paper has one genuinely strong contribution (C3A downstream correlation) and several interesting null results. The theoretical framework (Lotka-Volterra) is the paper's weakest element, not its strongest. The paper should be reframed around the downstream impact finding as the primary contribution, with the null results (LV, PMI) reported honestly as informative negative findings. The taxonomy needs fundamental methodological repair before publication. The current experimental base is insufficient for a top-tier venue but adequate for a workshop or short paper.

---

## Evidence Samples

### C3A key data point (strongest finding):
```
Absorption vs Sparse Probing F1: Pearson r = -0.5948, p < 0.001, n = 54
Partial r (controlling width, layer, arch) = -0.6611, p < 0.001
```

### C1B key data point (decisive failure):
```
LV detector test F1 = 0.128 at tau = 0.5
Cosine baseline test F1 = 0.165
F1 delta = -0.037 (LV is WORSE)
```

### C3B key data point (strong supporting evidence):
```
Low-absorption RAVEL (TPP): mean = 0.046
High-absorption RAVEL (TPP): mean = 0.009
Paired t-test: p = 0.006, Cohen's d = 2.13
```
