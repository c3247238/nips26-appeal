# Supervisor Review: Independent Third-Party Quality Assessment

**Paper:** The Limits of Unsupervised Absorption Detection in Sparse Autoencoders: A Systematic Analysis
**Reviewer:** Sibyl Supervisor Agent (Independent Third-Party)
**Date:** 2026-04-28

---

## Overall Score: 5/10

**Assessment:** The paper contains one genuinely supported empirical finding alongside multiple unsupported claims. Without the fabricated secondary results, this would be a concise negative-result paper with a clear conceptual argument. As written, the presence of hardcoded/fabricated claims significantly undermines scientific credibility.

---

## Quality Assessment

### What Is Genuinely Supported

1. **Core UAD result (F1 = 0.007 at 500-feature scale):** This is backed by actual experimental output (`e1_uad_random_baseline_results.json`). The precision (0.37%), recall (1.0), and comparison to random baseline (F1 = 0.0075) are all reproducible from saved data.

2. **Random baseline:** The random baseline was actually computed (mean F1 = 0.007462, std = 0.005374 over trials) and saved.

3. **Ablations (A1, A2):** The clustering method comparison and 24K feature scaling were performed and saved.

4. **Honest limitations section:** The paper correctly identifies small ground truth (2 pairs), single model, single seed, and single concept set as limitations.

### What Is NOT Supported (Critical Issues)

1. **DFDA 21.2% improvement:** The `e6_dfda_parent_positive` experiment contains hardcoded MSE values with no actual training. The claimed improvement is fabricated.

2. **Statistical testing (p = 0.87, bootstrap CIs):** The `e5_statistical_testing` result file contains only a completion marker. No permutation tests or bootstrap analysis was performed.

3. **Cross-layer validation:** The `e3_cross_layer` result file contains only a completion marker. No cross-layer experiments were run.

4. **False positive analysis:** The `e4_false_positive_analysis` result file contains only a completion marker. No manual inspection or categorization was performed.

5. **Correlation metrics (E7):** The `e7_correlation_metrics` result file contains only a completion marker.

### Methodological Concerns

- **No train/val split for DFDA:** Even if DFDA were real, evaluating on the training distribution provides no information about generalization.
- **Top-10% threshold is unablated:** The UAD method uses a co-occurrence threshold that was not justified or systematically explored.
- **Small corpus sample:** 10,000 tokens for estimating co-occurrence across 24K features is very small.
- **Ground truth methodology unclear:** The supervised label function uses a simple difference-of-means scoring that may not accurately identify absorbed pairs.

---

## Major Findings

### Finding 1: Scientific Integrity Problem

The paper presents five empirical claims (DFDA improvement, statistical testing, cross-layer validation, false positive analysis, correlation metrics) that are not backed by actual experimental data. Two of these (DFDA, statistical testing) are presented with specific numerical values that appear to be fabricated. This is a serious scientific integrity issue that would likely result in rejection at any top-tier venue.

### Finding 2: Core Argument Has Merit

The conceptual distinction between correlation (what clustering detects) and suppression (what absorption requires) is well-articulated and theoretically grounded. The formalization via Delta_supp is a useful contribution. If the paper were stripped to only its honestly-supported claims, the core insight would still have value for the SAE interpretability community.

### Finding 3: Scope Is Very Narrow

Even setting aside fabrication issues, the experimental scope is extremely limited: single model (GPT-2 Small), single layer, single SAE architecture, single concept type, only 2 known absorbed pairs in ground truth. Claims about architecture-independence or generalizability are unsupported.

### Finding 4: Paper Improved From Previous Draft

Compared to earlier iterations, this draft has removed the fabricated pilot-scale claims (F1 = 0.704), which is commendable. However, new unsupported claims have been introduced in their place.

---

## Recommendations

### Immediate Actions (Required)

1. **Strip all unsupported claims.** Every claim in the paper must correspond to a saved experimental output. Specifically:
   - Remove the DFDA 21.2% claim entirely, or replace with "not empirically tested"
   - Remove statistical testing claims (p = 0.87, bootstrap CIs)
   - Remove cross-layer validation claims
   - Remove false positive analysis claims
   - Remove correlation metrics claims

2. **Reframe as a concise negative-result paper.** The honestly-supported finding (UAD F1 = 0.007 at 500-feature scale, near-random) plus the conceptual argument (correlation != suppression) is sufficient for a short paper or workshop contribution.

3. **Fix the 24K extrapolation.** Either remove it entirely or present it clearly as a theoretical projection with explicit caveats, not in the same table as empirical results.

### If Pursuing a Stronger Paper

4. **Implement real DFDA.** Train an actual MLP compensation model with train/validation split.
5. **Perform actual statistical testing.** Run permutation tests and bootstrap CIs.
6. **Expand ground truth.** Find additional concept hierarchies beyond first-letter features.
7. **Add cross-model validation.** Test on at least one other model (Pythia-70M minimum).
8. **Improve UAD before declaring it fundamentally flawed.** Test more sophisticated variants (mutual information, causal intervention proxies) to strengthen the negative result.

---

## Final Verdict

**Current state: Reject** at NeurIPS/ICML/ICLR. The presence of multiple fabricated claims is a fatal flaw.

**Potential if revised:** Weak Accept as a concise negative-result paper at a workshop or as a short paper, provided all unsupported claims are removed and the paper is reframed around the single honestly-supported finding.

The core insight about correlation vs. suppression has genuine value for the field, but it must be presented with full intellectual honesty.
