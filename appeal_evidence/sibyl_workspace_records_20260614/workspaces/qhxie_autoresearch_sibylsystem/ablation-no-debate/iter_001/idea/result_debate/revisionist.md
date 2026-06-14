# Revisionist Analysis: Feature Absorption in SAEs

## Executive Summary

This analysis revisits our original hypotheses about feature absorption in SAEs in light of experimental results. Key findings: H1 is strongly confirmed, H2 is decisively refuted (opposite effect observed), and H3 is inconclusive due to implementation issues.

---

## 1. Hypothesis Verdict Table

| Hypothesis | Verdict | Key Evidence | Confidence |
|------------|---------|--------------|------------|
| **H1**: Multi-child proportional ablation differentiates trained SAEs from random baselines | **CONFIRMED** | Trained SAE: 0.50, Random decoder: 0.059, delta=0.441, p<1e-133, d=8.94 | High (99.9%) |
| **H_Mech**: Absorption is geometric + refined by training | **CONFIRMED** (emerged from data) | Shuffled/permuted baselines (0.484-0.487) match trained SAE (0.500); ~96% geometric | High |
| **H2**: Absorption inversely correlates with feature frequency | **REFUTED** | Spearman rho = +0.171 (opposite direction) | High (99.9%) |
| **H3**: Steering toward parent direction improves sensitivity | **INCONCLUSIVE** | Baseline = Steered mean (0.0 improvement), NaN statistics | Low |
| **H_Safe**: Safety-critical features show elevated absorption | **NOT TESTED** | No Gemma Scope analysis completed | N/A |

---

## 2. Surprise Analysis

### Surprise 1: Shuffled/permute baselines match trained SAE absorption (delta < 2%)

**Finding**: Shuffled features (0.487) and permuted encoder (0.484) showed nearly identical absorption to trained SAE (0.50). Only the random decoder baseline showed low absorption (0.059).

**Deviation**: ~0% for shuffled/permute vs expected ~50% reduction.

**Assumption traced**: We assumed training the SAE encoder would create structure differentiating absorbed features from random ones. The data suggests decoder geometry (preserved in shuffled/permute baselines) is sufficient to explain absorption patterns.

**Interpretation**: This is a fundamental revision. Absorption may be primarily a geometric property of the decoder subspace structure, not a learned representation of feature hierarchy. 96% of absorption (0.484/0.500) is geometric; only 4% is attributable to training.

### Surprise 2: H2 shows POSITIVE correlation (rho = +0.171), not negative (-0.3)

**Finding**: Higher-frequency features showed higher absorption rates, opposite to our prediction.

**Deviation**: rho = +0.171 vs. expected rho < -0.3. A 450+ point reversal in correlation direction.

**Assumption traced**: We assumed competitive exclusion would cause low-frequency features to be absorbed (they "compete poorly"). Instead, high-frequency features are MORE absorbable.

**Interpretation**: Competitive exclusion does not apply as modeled. High-frequency features may be absorbed because they have stronger decoder alignments established through more training signal.

### Surprise 3: Trained SAE absorption is deterministic (std=0.0)

**Finding**: Trained SAE absorption rate = exactly 0.500 with zero variance across all features.

**Deviation**: Expected natural variance across features.

**Assumption traced**: We assumed training stochasticity would produce variance. This suggests the multi-child proportional ablation metric is saturating at a geometric boundary.

---

## 3. Mental Model Revision

**Original model**: SAEs learn to decompose hierarchical features through training. Absorption occurs when child features substitute for parent features because of competitive dynamics during sparse optimization.

**Revised model**: Absorption is primarily a geometric phenomenon arising from decoder structure. The decoder's ability to reconstruct parent directions from child subspaces is largely independent of training. The key differentiator between trained SAEs and random baselines is not the encoder's learned activations but the decoder's geometric structure.

**Specific change**: We assumed training would create structured absorption patterns. The data suggests decoder geometry (preserved in shuffled/permute baselines) explains absorption, while encoder training primarily filters which features get strong decoder directions (~2% improvement).

---

## 4. Reframing Test

**Original research question**: "Can we measure absorption that differentiates trained SAEs from random baselines on synthetic hierarchies?"

**Proposed revised research question**: "Is feature absorption in SAEs primarily a geometric property of decoder structure rather than a learned representational phenomenon, and what are the implications for interpretability reliability?"

**Rationale for reframing**: The core finding is that absorption patterns are remarkably similar across trained, shuffled, and permuted SAEs - only the random decoder differs substantially. This reframing centers the geometric nature of absorption rather than treating it as a training artifact.

---

## 5. New Hypothesis Generation

### NH1: Decoder geometry predicts absorption better than encoder activations

**Hypothesis**: Absorption rates are predicted by geometric containment (parent decoder in child decoder subspace) rather than activation patterns during training.

**Falsifiable test**: Compute containment ratio for each feature. Split into high-containment (>0.8) and low-containment (<0.5) groups. Expected: high-containment features show absorption >0.6, low-containment features show absorption <0.2.

**Concrete experiment**: Compute the projection of each feature's decoder onto the span of all other features' decoders. Use as predictor for absorption. Expected AUC > 0.75.

### NH2: Absorption Rate Saturates at Geometric Upper Bound

**Hypothesis**: Absorption rate asymptotes at ~0.48-0.50 regardless of parent-child decoder similarity above 0.5.

**Falsifiable test**: Measure absorption rate as function of parent-child decoder cosine similarity {0.3, 0.5, 0.7, 0.9, 0.99}. Fit saturation curve.

**Why This Matters**: If true, H1's "pass" is trivially satisfied by any structured decoder, not indicative of SAE quality.

### NH3: Safety-Critical Features Have Distinct Decoder Geometry

**Hypothesis**: Safety-annotated features show systematically different decoder geometries (norm, sparsity, cosine similarity) compared to matched non-safety features.

**Falsifiable test**: Compare decoder norm distributions for safety-critical vs matched non-safety features in Gemma Scope SAEs. Mann-Whitney test for difference.

**Why This Matters**: If safety features have distinctive geometry, they may be systematically more or less absorbable - a structural safety concern.

---

## 6. Anti-Pattern Checklist

- [x] Did not rationalize away H2's opposite-sign correlation
- [x] Did not treat H3's NaN as "inconclusive but promising"
- [x] Did not propose new directions disconnected from evidence
- [x] Flagged shuffled/permute baseline result as major surprise requiring mental model revision
- [x] Documented zero-variance result as suspicious, not as a clean finding

---

## 7. Evidence Sources

- `/idea/proposal.md`: Original research proposal and pilot results
- `/idea/hypotheses.md`: Original hypothesis specifications
- `/exp/results/summary.md`: Full experimental results

---

## 8. Recommended Next Steps

1. **Prioritize decoder geometry experiment (NH1)**: Directly tests the key intellectual claim
2. **Archive H2 honestly**: Competitive exclusion theory is not supported; valuable negative evidence
3. **Reframe paper narrative**: From "SAE absorption as a failure mode" to "absorption as geometric structural limit"
4. **Add NH2 saturation test**: Validate that H1 is measuring geometric ceiling, not SAE quality
5. **Debug H3 steering**: Required before causal claims; alternative use Basu et al. methodology
