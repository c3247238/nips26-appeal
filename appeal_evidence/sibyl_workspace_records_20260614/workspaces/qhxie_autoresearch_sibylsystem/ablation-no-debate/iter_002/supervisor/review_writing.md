# Supervisor Review: Feature Absorption in SAEs

**Overall Score: 6 (Borderline Reject)**
**Verdict: Revise**

## Executive Summary

The paper makes a genuine methodological contribution in introducing the factorial decomposition for isolating encoder vs decoder contributions to absorption. The encoder dominance finding (80x larger than decoder effect) and the capacity-pressure mechanism (lower L0 -> higher absorption) are real and interesting findings. However, three critical weaknesses undermine the strength of these contributions:

1. **Post-hoc criterion revision** for H_Mech destroys confirmatory status and renders the primary result exploratory
2. **Metric inconsistency** across experiments prevents reliable cross-experiment comparison  
3. **Synthetic-only validation** at d_model=128 provides limited evidence for real-world LLM SAE behavior

The negative results (steering, safety) are honestly reported, but the safety analysis uses a saturating metric (97% absorption for both groups) that makes the null result uninterpretable.

---

## Dimension Scores

| Dimension | Score | Justification |
|-----------|-------|---------------|
| **Novelty** | 7 | The factorial decomposition method is genuinely novel as a measurement technique. However, the underlying finding (encoder drives absorption) is consistent with Oursland (2026)'s theoretical prediction, and SAEBench (2025) already noted sparsity differences across architectures. Contribution is empirical quantification, not discovery. |
| **Technical Soundness** | 5 | The methodology is sound but post-hoc criterion revision undermines confirmatory status. The capacity-pressure mechanism is post-hoc and not distinguished from TopK artifact. Multiple theoretical claims lack formal justification. |
| **Experimental Rigor** | 5 | Key experiments have design flaws: H_Mech uses post-hoc criteria, H_Safe uses saturating metric (97%), held-out generalization has n=1 per seed. Multi-seed validation shows suspicious zero-variance in random baselines. |
| **Reproducibility** | 5 | d_model=128 synthetic data limits real-world generalization. No code released for hierarchy generator or absorption measurement. No validation at scale. |

---

## Critical Issues

### 1. Post-hoc Criterion Revision (Critical)

The original H_Mech pass criteria (B approx D and C approx A) failed on 14 of 15 runs (93.3%). Criteria were revised to encoder effect > 0.5 and decoder effect < 0.1 **after observing the data**. This revision was not pre-registered.

The paper acknowledges this should be "interpreted as exploratory rather than confirmatory" -- but the entire H_Mech contribution (the paper's primary empirical claim) rests on this revised criterion. A NeurIPS reviewer would likely flag this as a significant weakness requiring either pre-registration or reframing as exploratory work.

### 2. Metric Inconsistency (Critical)

Three different absorption metrics are used without established equivalence:
- Factorial decomposition: "overlap fraction"
- Multi-seed validation: "Jaccard overlap"  
- Safety analysis: "cosine-based proportional absorption"

Table 1 presents results from all three metrics as if comparable, which they are not. The paper cannot legitimately claim "encoder effect is 80x larger than decoder effect" on a unified scale when different metrics were used.

### 3. Random Baseline Zero Variance (Major)

Random SAE baselines show std ~0 across all 5 seeds in multiseed_validation.json. This is highly suspicious -- either the random SAE produces identical patterns regardless of seed, or the measurement has no sensitivity to the random SAE's actual state. Cohen's d > 10 is mathematically misleading when one distribution has zero variance.

### 4. Safety Analysis Saturates (Major)

Both safety and non-safety groups show ~97% absorption (0.967 vs 0.968). At 97%, there is no dynamic range to detect differences. The Mann-Whitney p=0.989 null result is uninterpretable -- the metric cannot distinguish groups even if real differences existed.

### 5. Held-Out Generalization n=1 (Major)

The r=0.998 correlation is based on only 5 data points (n=1 held-out hierarchy per seed, 5 seeds). With 3 degrees of freedom, the confidence interval could be wide despite the near-perfect point estimate. The paper presents this as "perfect generalization" without reporting the CI.

---

## What Would Raise the Score

1. **Pre-register the H_Mech revised criteria** and replicate with those criteria as the official stopping rule (changes status from exploratory to confirmatory)
2. **Add d_model=512 validation** to establish whether encoder dominance scales beyond pilot-scale experiments
3. **Replace the saturating safety metric** with one that has dynamic range to detect differences
4. **Release synthetic hierarchy generator and absorption measurement code** for reproducibility
5. **Add diagnostic** distinguishing capacity-pressure mechanism from TopK artifact (compare with JumpReLU SAE)

---

## Summary Assessment

This paper has genuine merit -- the factorial decomposition method is novel, the encoder dominance finding is real and interesting, and the negative results are honestly reported. However, the combination of post-hoc criteria revision, metric inconsistency, and synthetic-only validation at small scale creates a gap between the paper's claims and the evidence supporting them. With another iteration to address these issues, this could be a strong paper. As submitted, it sits at the borderline.

**Recommendation**: Revise to address the critical issues before resubmission.
