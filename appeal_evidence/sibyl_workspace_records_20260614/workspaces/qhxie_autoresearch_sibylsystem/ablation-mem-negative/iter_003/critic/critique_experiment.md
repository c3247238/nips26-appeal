# Experiment Critique: Methodological and Statistical Review

> Critic: sibyl-critic (heavy tier)
> Date: 2026-04-29
> Scope: All experiment results (pilots + full) + methodology.md

---

## Overall Experiment Assessment

**Strength**: The experiments are well-structured, with clear hypotheses, pre-registered criteria, and honest reporting of negative results. The ablation study is comprehensive.

**Weakness**: Critical methodological issues around ground truth definition, sample size, circular metrics, and missing controls undermine the strength of the conclusions.

---

## 1. Ground Truth Definition (Critical)

### The Problem

The paper defines "true absorption rate" as:

```
R_abs(c_i, c_j) = |A(c_i) ∩ A(c_j)| / |A(c_i) ∪ A(c_j)|
```

where A(c) is the set of top-K features by mean activation on tokens belonging to c.

And defines "collision rate" as:

```
R_collision(c_i, c_j) = |T(c_i) ∩ T(c_j)| / |T(c_i) ∪ T(c_j)|
```

where T(c) is the set of top-K features by mean activation on tokens belonging to c.

**These are the same metric.** The correlation between them is not proxy validation; it is self-correlation.

### Evidence of the Problem

From `f4_collision_correlation_results.json`, the pair "one-two" has:
- collision_rate: 0.25
- true_absorption: 0.4

But both are computed from the same shared_features list ([13586, 10972]). The only difference is the denominator (overlap vs Jaccard). This is not two independent measurements.

### The Feature 11746 Problem

The pilot summary explicitly notes: "The top-5 features are nearly identical across all 26 letters (feature 11746 dominates), suggesting this SAE has a strong 'first-letter' super-feature."

This means the GT definition may be detecting "super-features" (single features that activate on many related tokens) rather than true distributed absorption. The paper never addresses this concern.

### Fix Required

1. Acknowledge that the operational definition of absorption is an approximation
2. Distinguish between the conceptual definition (Chanin et al.'s probe-based method) and the operational definition (top-K overlap)
3. Frame collision rate validation as "reliability of the operational definition" not "proxy validation"
4. Discuss the super-feature concern explicitly

---

## 2. Sample Size and Statistical Power (Critical)

### UAD Evaluation

- **Ground truth pairs**: 7
- **Detected pairs**: 4,155
- **True positives**: 1

With only 7 positives, the F1 score has enormous variance. A single additional TP would change F1 from 0.00048 to ~0.00096 (a 100% increase). The paper presents F1 = 0.00048 as definitive when it is actually extremely noisy.

### Confidence Intervals Missing

The paper reports exact F1 values without confidence intervals. For a proportion estimate with 1 success out of 4155 trials, the 95% CI for precision is approximately [0.00006, 0.0014] --- three orders of magnitude wide.

### Fix Required

1. Add bootstrap confidence intervals for F1, precision, and recall
2. Acknowledge that the UAD evaluation is based on a tiny sample
3. Frame conclusions as "UAD fails on this test set" rather than universal claims

---

## 3. The F1 Identity Problem (Major)

### The Issue

From `f2_uad_ablations_results.json`:

| Method | TP | FP | FN | F1 |
|--------|-----|------|-----|--------|
| Full UAD | 1 | 4154 | 6 | 0.0004805 |
| Same-cluster random | 1 | 4154 | 6 | 0.0004805 |

These are IDENTICAL because both have exactly the same TP, FP, and FN counts. This is not an empirical finding that "UAD equals random" --- it is a mathematical identity. The paper presents this as a statistical result.

### Why This Happened

The same-cluster random baseline was designed to sample the same number of pairs (4155) from within the same clusters. By construction, it will have the same number of detected pairs. Since there is only 1 GT pair in any cluster, both methods detect exactly 1 TP.

### Fix Required

Reframe: "UAD detects 1 true positive out of 4,155 candidates. Randomly sampling 4,155 pairs from the same cluster structure yields the same 1 true positive by chance. UAD's pipeline provides zero improvement over this trivial baseline."

---

## 4. K-Means Ablation: Missed Opportunity (Major)

### The Result

From `f2_uad_ablations_results.json`:

| Variant | TP | FP | Recall | F1 |
|---------|-----|------|--------|--------|
| K-means | 6 | 3237 | 85.7% | 0.0037 |
| Full UAD | 1 | 4154 | 14.3% | 0.00048 |

K-means achieves **6x the recall** of full UAD. This is a significant difference that the paper dismisses without analysis.

### The Question Not Asked

Why does K-means succeed where Ward linkage fails? Possible explanations:
1. K-means groups features by some non-co-occurrence property (e.g., activation magnitude)
2. K-means happens to place GT features in the same cluster by chance more often
3. The phi vectors have structure that K-means captures better than hierarchical clustering

Understanding WHY K-means performs better would strengthen the paper's constructive contribution.

### Fix Required

Add analysis: compare the cluster assignments of GT features under K-means vs Ward linkage. What property does K-means use to group them? Could a post-hoc threshold on cluster coherence improve precision?

---

## 5. Ablation Filter No-Ops (Minor)

### The Issue

From `f2_uad_ablations_results.json`:

| Variant | TP | FP | FN |
|---------|-----|------|-----|
| Full UAD | 1 | 4154 | 6 |
| No dead filter | 1 | 4154 | 6 |
| No phi filter | 1 | 4154 | 6 |

Removing dead feature filtering and phi coefficient filtering produces IDENTICAL results. This suggests:
1. Dead feature filtering removed zero features (all 500 passed the variance threshold)
2. Phi filtering removed zero pairs (all pairs passed the threshold)

The paper does not report how many features/pairs were actually filtered.

### Fix Required

Report the number of features removed by each filtering step. If zero were removed, explain why (e.g., the variance threshold was too permissive for this SAE).

---

## 6. Single Seed, No Sensitivity Analysis (Minor)

All experiments use seed 42. While token-level mutual exclusivity is deterministic (feature 11513 will always fire on "three"), the following may vary with seed:
- Corpus sample selection (1,000 sequences from OpenWebText)
- Feature selection (500 most active features)
- Clustering results ( Ward linkage is deterministic, but K-means is not)

### Fix Required

Run UAD with 2-3 additional seeds and report variance in F1. This is a quick check that would either strengthen or weaken the conclusion.

---

## 7. Missing Controls

### Control 1: Semantic Hierarchy

The paper tests numbers and punctuation (token-disjoint hierarchies) but not semantic hierarchies (animal/dog) where children MAY co-occur. Testing at least one semantic hierarchy would:
- Strengthen the universal claim about mutual exclusivity (if it holds)
- Reveal a boundary condition (if it fails)

### Control 2: Different SAE Architecture

All experiments use gpt2-small-res-jb (a JumpReLU SAE). Testing a TopK or Gated SAE would test whether the findings generalize.

### Control 3: Different Layer

Layer 8 only. Different layers may have different feature structures.

### Fix Required

Add these as explicit limitations. Consider running at least one additional test (e.g., a semantic hierarchy or different layer) if time permits.

---

## 8. False Positive Rate Terminology (Minor)

From `f5_false_positive_results.json`:
- "fp_rate": 0.9997593261131167

This is computed as FP/(TP+FP) = 4154/4155, which is 1 - precision.

In standard ML terminology, "false positive rate" means FP/(FP+TN), which would be much lower (4154 / (4154 + ~122,000) ≈ 3.3%).

Using non-standard terminology without clarification is confusing.

### Fix Required

Use "precision = 0.024%" instead of "false positive rate = 99.98%."

---

## 9. Collision Correlation: Hierarchy-Specific Results

From `f4_collision_correlation_results.json`:

| Hierarchy | N | Spearman r | p-value |
|-----------|---|-----------|---------|
| Numbers | 28 | 0.598 | 0.00078 |
| Punctuation | 28 | 0.693 | 4.3e-05 |
| Combined | 56 | 0.869 | 4.2e-18 |

The combined correlation (0.869) is HIGHER than either individual hierarchy. This is unusual---combining two datasets with correlations ~0.6-0.7 typically yields a correlation in that range, not higher.

The explanation: the two hierarchies have different mean collision rates (numbers: 0.387, punctuation: 0.051) and different mean absorption rates (numbers: 0.668, punctuation: 0.119). Combining them creates a bimodal distribution that inflates the correlation.

### Fix Required

Acknowledge this statistical artifact. Report both individual and combined correlations, noting that the combined correlation benefits from between-hierarchy variance.

---

## 10. Reproducibility Checklist

The methodology claims "All experiments completed with fixed random seed, exact model versions, dataset sample count." However:

- [x] Fixed random seed (42)
- [x] Exact model/SAE versions recorded
- [x] Dataset sample count recorded
- [ ] **Code repository link**: MISSING
- [ ] **Pre-registration**: Hypotheses documented but not pre-registered
- [ ] **Annotation protocol**: GT pairs identified without explicit criteria
- [ ] **Sensitivity analysis**: Missing

---

## Summary: Experiment Quality Score

| Dimension | Score | Notes |
|-----------|-------|-------|
| Hypothesis testing | 7/10 | Pre-registered criteria, but poorly calibrated |
| Ground truth quality | 4/10 | Circular definition, super-feature concern |
| Sample size | 3/10 | 7 GT pairs is inadequate for reliable F1 |
| Ablation design | 7/10 | Comprehensive but filter no-ops unexplained |
| Statistical methods | 5/10 | Missing CIs, FPR terminology confusion |
| Controls | 4/10 | Missing semantic hierarchy, different SAEs |
| Reproducibility | 5/10 | No code link, no sensitivity analysis |
| Honesty | 9/10 | Excellent negative result reporting |
| **Overall** | **5.5/10** | Honest but methodologically flawed |
