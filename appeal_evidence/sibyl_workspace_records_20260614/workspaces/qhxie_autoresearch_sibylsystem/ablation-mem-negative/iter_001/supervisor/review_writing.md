# Supervisor Review: Feature Absorption in Sparse Autoencoders

**Reviewer**: Independent third-party supervisor (NeurIPS/ICML/ICLR calibrated)
**Score**: 5.5 / 10 (Borderline Reject)
**Verdict**: CONTINUE (requires substantial revision)

---

## Executive Summary

This paper presents a cross-architecture study of feature absorption/collision in Sparse Autoencoders (SAEs) on GPT-2 Small. The manuscript's strongest and most admirable asset is its **exemplary honest reporting of negative results**---H2, H3, and H4 are all falsified, and the paper reports this without spin. This scientific integrity is rare and commendable.

However, the core contribution is severely undermined by methodological flaws that would likely trigger rejection at a top-tier venue:

1. **Massively confounded architecture comparison**: The headline 4x collision rate difference (15.4% vs. 3.8%) compares a pretrained JumpReLU (Gemma data, d_SAE=24,576) against a trained TopK (OpenWebText, d_SAE=16,384). Architecture, training data, dictionary size, and training procedure are all confounded. The paper acknowledges this in Section 4.2 but still leads with the 4x difference in the Abstract and Conclusion.

2. **Catastrophic training failure**: 89-99% dead feature ratios across all trained SAEs mean the effective dictionary is <1% of nominal size. At k=10, only ~100 features are active out of 16,384. This likely invalidates all trained-SAE results.

3. **Severely underpowered designs**: All three primary hypotheses (H2-H4) yield near-zero correlations with n=5-6 data points and ~20% statistical power. The null results are consistent with both "no effect" and "insufficient power," but the paper leans toward the former without adequate qualification.

4. **Weak proxy metric**: Collision rate (multiple concepts sharing a feature) is not true absorption (parent suppressing child under co-occurrence). The entire CAAB benchmark is built around this proxy without validating that collision correlates with absorption.

5. **Underpowered exploratory methods**: UAD (25 collisions) and DFDA (4 pairs, all sharing feature 18486) are proof-of-concepts framed as contributions.

**Bottom line**: The paper's actual contribution may be limited to a narrow negative result on first-letter features. This does not meet the bar for NeurIPS/ICML/ICLR in its current form.

---

## Dimension Scores

### 1. Novelty & Significance: 5/10

The core claim---that collision rate differs by architecture and may be a poor proxy---is weakened by the confounded comparison. If the 4x difference is driven by training data/width rather than architecture, the "cross-architecture benchmark" contribution evaporates.

Chanin et al. (2024) already documented absorption; this paper's contribution is measuring a proxy metric (collision) across architectures. If the comparison is confounded, the contribution is incremental at best.

The honest negative result reporting (H2-H4 falsified) is valuable and the paper's strongest aspect. However, negative results alone---especially on a narrow task (first-letter probing) with severe methodological limitations---may not be sufficient as a standalone contribution for a top-tier venue.

**What would raise this score**: A matched architecture comparison with validated collision-absorption correlation, plus expansion to multiple concept domains.

### 2. Technical Soundness: 5/10

The collision rate metric is not validated against true absorption. A feature shared by 5 letters (feature 18486: c, i, o, p, u) may simply be polysemantic, not absorbed. Without parent-child hierarchy testing (Chanin et al. protocol), calling this "absorption" is misleading.

The "causal" claim in E2 is weak: 5 k-values from the same training run with different k is a parameter sweep, not a causal design. k directly controls both sparsity and reconstruction quality, creating a structural confound.

The specificity score metric is numerically unstable (mean specificity 6.56e8 for JumpReLU, driven by division-by-near-zero). This metric should not be reported.

Post-hoc power analysis (Section 3.6) is methodologically questionable. Power should be computed before the experiment, not after.

**What would raise this score**: Validate collision against true absorption rate, replace post-hoc power with confidence intervals, and reframe E2 as a parameter sweep.

### 3. Experimental Rigor: 6/10

**Strengths**:
- Honest reporting of negative results (H2-H4 all falsified) with specific expected vs. observed values
- All numbers in tables are verifiable against source JSON files
- Clear experimental protocol with fixed seed and documented hyperparameters

**Weaknesses**:
- Single seed (42) with no replication. The pilot showed 30.8% collision for TopK while the full experiment shows 3.8%---an 8x discrepancy that is never explained.
- n=5-6 data points for all correlation analyses; ~20% power to detect medium effects
- 89-99% dead feature ratios suggest the SAEs are not functioning as intended
- No validation on concept domains other than first letters
- UAD validated on only 25 true collisions; DFDA on only 4 pairs
- No statistical test distinguishing "no effect" from "insufficient power"

**Cross-validation against raw data**: All reported numbers match the source JSON files exactly. The writing review verified every value in every table. This is commendable data integrity.

**What would raise this score**: Multi-seed replication (3+ seeds), reduce dead features below 50%, expand to additional concept domains, and add equivalence testing.

### 4. Reproducibility: 5/10

The paper documents hyperparameters, training details, and seed. However:
- Single seed means results may not replicate with different seeds
- The 8x pilot/full discrepancy raises serious concerns about measurement stability
- Pretrained JumpReLU from GemmaScope may not be reproducibly loadable (API issues noted)
- Dead feature ratios of 89-99% suggest training configuration is unstable
- No code repository URL ("[anonymous repository]" placeholder)
- UAD validation depends on supervised labels from first-letter features, which are domain-specific

**What would raise this score**: Multi-seed replication, code release, and explanation of the pilot/full discrepancy.

---

## Issue Classification

### Critical Issues (would cause rejection on their own)

1. **Confounded architecture comparison** (Experiment): The 4x collision rate difference cannot be attributed to architecture. The paper leads with this uninterpretable result.

2. **Catastrophic dead feature ratios** (Experiment): 89-99% dead features likely invalidates all trained-SAE results. This is not a minor confound---it's a training failure.

### Major Issues (significantly weaken the paper)

3. **Underpowered designs with overinterpreted null results** (Analysis): n=5-6 with single seed; null results are inconclusive, not conclusive evidence.

4. **Collision rate is an unvalidated proxy** (Methodology): No correlation with true absorption rate is reported. The benchmark may measure the wrong thing.

5. **E2 is not causal** (Experiment): Parameter sweep masquerading as causal inference.

6. **UAD/DFDA are severely underpowered** (Experiment): 25 collisions and 4 pairs are not sufficient for contribution claims.

7. **Single seed, no replication** (Reproducibility): Results may not generalize to other seeds.

8. **Limited novelty** (Novelty): Core contribution may evaporate if confounds are fixed.

### Minor Issues (should be fixed but don't affect verdict)

9. Abstract omits "per-pair residual" qualifier for DFDA claim.
10. Post-hoc power analysis is misleading.
11. Gemma-2-2B fallback is buried in Section 3.3.
12. Concept set limited to 26 first-letter features.
13. Non-monotonic sparsity pattern may be noise.
14. Conclusion overgeneralizes from first-letter probing to "absorption harm."

---

## Risks

1. **Reviewer fatal flaw**: Any competent reviewer will flag the confounded architecture comparison as uninterpretable.
2. **Training failure**: The 89-99% dead feature ratios suggest the trained SAEs are not functioning; reviewers may dismiss all trained-SAE results.
3. **Replication failure**: Single seed with n=5-6 means results may not replicate with different seeds.
4. **Misleading framing**: The collision/absorption terminology confusion may be seen as intentionally inflating relevance.
5. **Trivial UAD**: 100% recall with 54.3% precision is achievable with a permissive threshold; reviewers may dismiss it.
6. **Metric gaming**: DFDA's 11.1% on 10^-6 scale MSE is negligible in absolute terms.

---

## Evidence Gaps

1. No correlation between collision rate and true absorption rate (Chanin et al. protocol).
2. No multi-seed replication for any experiment.
3. No validation on concept domains other than first letters.
4. No explanation for the 8x pilot/full collision rate discrepancy (30.8% vs 3.8%).
5. No investigation of why dead feature ratios are catastrophically high.
6. No matched-architecture comparison (same d_SAE, same training data, same procedure).
7. No code repository URL.
8. No statistical test distinguishing "no effect" from "insufficient power."

---

## What Would Raise the Score

**To 6.5 (Borderline Reject -> Weak Accept)**:
- Fix the dead feature problem (reduce below 50% via longer training or hyperparameter tuning)
- Run multi-seed replication (3+ seeds) for E1-E3 with standard errors
- Explain the pilot/full discrepancy

**To 7.5 (Borderline Accept)**:
- All of the above, plus:
- Implement matched architecture comparison (same d_SAE, same data, both trained from scratch)
- Validate collision rate against true absorption rate on parent-child hierarchies
- Add at least one more concept domain

**To 8.0+ (Accept)**:
- All of the above, plus:
- Demonstrate UAD generalizes beyond first letters
- Show DFDA improves overall reconstruction MSE or semantic recovery via probe accuracy
- Run experiments on a second model (e.g., Pythia-1.4B)

---

## Acknowledgment of Strengths

Despite the low score, the paper has genuine strengths that should be preserved:

1. **Honest negative result reporting**: H2-H4 falsified, reported without spin. This is the paper's most valuable asset and should be retained in any revision.
2. **Verifiable data integrity**: Every number in every table matches the source JSON files exactly.
3. **Clear structure**: RQ-contribution-experiment mapping is easy to follow.
4. **Exemplary limitation disclosure**: Section 5.5 openly lists 5 major limitations with future work mapping.

These strengths are why the score is 5.5 rather than 4.0. With substantial methodological fixes, this could become a solid contribution. Without them, it will likely be rejected.
