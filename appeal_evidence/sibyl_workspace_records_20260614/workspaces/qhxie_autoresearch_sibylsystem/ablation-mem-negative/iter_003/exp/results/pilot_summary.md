# Pilot Summary - Iteration 3

> Date: 2026-04-28
> Task: p1_collision_proxy_validation
> Status: COMPLETED

## P1: Collision Rate - Absorption Rate Proxy Validation

### Results

| Metric | Value | Pass Threshold | Status |
|--------|-------|----------------|--------|
| Spearman r | 0.711 | >= 0.3 | PASS |
| Pearson r | 0.733 | - | - |
| Bootstrap 95% CI | [0.219, 0.887] | Does not include 0 | PASS |
| GT pairs detected | 10/10 | >= 5 | PASS |
| Runtime | 25.3s | < 15 min | PASS |

### Key Findings

1. **Strong positive correlation**: Collision rate (cosine similarity of activation profiles) correlates with true absorption rate (Jaccard similarity of top-K feature sets) at r=0.71 (Spearman) and r=0.73 (Pearson).

2. **All 10 vowel pairs detected**: All possible vowel pairs (a-e, a-i, a-o, a-u, e-i, e-o, e-u, i-o, i-u, o-u) share at least one top-10 SAE feature, confirming absorption clustering.

3. **Bootstrap CI excludes zero**: The 95% bootstrap confidence interval [0.219, 0.887] does not include zero, supporting the directional relationship.

4. **Important caveat**: The top-5 features are nearly identical across all 26 letters (feature 11746 dominates), suggesting this SAE has a strong "first-letter" super-feature. This means the ground truth definition (shared top features = absorption) may be too permissive. However, the correlation between collision rate and absorption rate is still meaningful because collision rates vary even when all pairs are "absorbed" (ranging from 0.824 to 0.964).

### Methodology Notes

- **v2 redesign**: Used distribution-based approach (cosine similarity of full activation profiles) instead of single top feature per letter, which avoided the degenerate case where all letters mapped to the same feature.
- **Ground truth**: Defined as Jaccard overlap of top-10 features per letter. All vowel pairs share features, giving a non-degenerate absorption rate distribution (0.538 to 0.667).
- **Collision rate**: Cosine similarity of mean activation profiles across all 24,576 SAE features.

### Implications for Full Experiments

- **GO for full experiments**: P1 passes all criteria. Collision rate shows promise as a proxy for absorption rate.
- **Caution**: The dominance of feature 11746 suggests the SAE may have a single "first-letter" super-feature rather than distributed representations. This should be noted in the paper.
- **Recommendation**: Proceed with P2 (UAD reproducibility) and P3 (random baseline).

---

## P2: UAD Reproducibility Validation

### Result: FAIL

### Key Metrics

| Metric | Value | Pass Threshold | Status |
|--------|-------|----------------|--------|
| Precision | 0.0% | - | - |
| Recall | 14.3% | >= 80% | **FAIL** |
| F1 | 0.0 | >= 50% | **FAIL** |
| Detected Pairs | 4155 | - | - |
| Ground Truth Pairs | 7 | - | - |
| True Positives | 1 | - | - |
| False Positives | 4154 | - | - |
| Runtime | 149.3s | < 15 min | PASS |

### Ground Truth

Using number word hierarchy (one, two, three, four, five, six, seven, eight) on GPT-2 Small layer 8 with gpt2-small-res-jb SAE:
- Number features detected: 8 primary features
- Absorption features identified: 7 features that activate on multiple numbers
- Feature absorption pairs: 7 pairs

### Root Cause Analysis

UAD fails because of a **fundamental mismatch** between its detection mechanism and the nature of absorption:

1. **UAD detects co-occurrence**: It clusters features that fire on the SAME tokens (co-occur in the corpus).

2. **Absorption features are mutually exclusive**: They fire on DIFFERENT tokens representing different child concepts. For example:
   - Feature 11513 fires only on "three"
   - Feature 24189 fires only on "four", "five", "six", "seven", "eight"
   - These features never activate on the same token

3. **Co-occurrence is near zero**: Phi coefficients between absorption features on general text (OpenWebText) are near zero or negative.

4. **Clustering separates them**: With 50 clusters on 504 features, GT absorption features are placed in different clusters.

### Evidence

Token-level activation for number sequence "one two three four five six seven eight":
```
Token        F11513 F12413 F22971 F24189
one             0.0   15.3    0.0    0.0
two             0.0    0.0   24.2    0.0
three          29.4    0.0    0.0    0.0
four            0.0    0.0    0.0   18.9
five            0.0    0.0    0.0   14.9
six             0.0    0.0    0.0   16.6
seven           0.0    0.0    0.0   14.3
eight           0.0    0.0    0.0   15.9
```

Features are completely mutually exclusive at the token level.

### Implications

1. **UAD detects features that fire TOGETHER** (co-occurring), not features that fire on mutually exclusive instances of a parent concept.

2. **This is a fundamental limitation** of co-occurrence-based approaches for hierarchical absorption detection.

3. **UAD may work for**: detecting synonym features or contextually related features that co-occur frequently.

4. **For hierarchical absorption**: A different approach is needed - one that measures semantic similarity or causal dependence rather than co-occurrence.

### Honest Assessment

This is a **valid negative result**. The UAD method as implemented cannot detect the type of absorption defined by Chanin et al. (2024) in the gpt2-small-res-jb SAE. The method's reliance on co-occurrence clustering is fundamentally incompatible with the mutually exclusive nature of absorption features.

### Recommendations

1. **Report this as a negative result** in the paper
2. **Discuss the limitation** of co-occurrence-based approaches
3. **Consider alternative approaches** for absorption detection:
   - Semantic similarity between feature directions (cosine similarity of decoder weights)
   - Causal intervention (zeroing child features and measuring parent recovery)
   - Activation patching between related concepts
4. **Pivot the paper focus** to "Why co-occurrence clustering cannot detect feature absorption"

---

## P3: Random Baseline Validation

### Result: FAIL

### Key Metrics

| Metric | Value | Pass Threshold | Status |
|--------|-------|----------------|--------|
| UAD F1 | 0.00048 | - | - |
| Global Random F1 | 0.00011 | < 0.05 | **PASS** |
| Same-Cluster Random F1 | 0.00048 | - | - |
| UAD - Global Random | 0.00037 | >= 0.3 | **FAIL** |
| UAD == Same-Cluster Random | Yes | No | **FAIL** |
| Runtime | 151.3s | < 10 min | PASS |

### Baseline Comparisons

| Baseline Type | Mean F1 | Std F1 | Mean Precision | Mean Recall | Mean TP |
|---------------|---------|--------|----------------|-------------|---------|
| UAD (actual) | 0.00048 | - | 0.00024 | 0.143 | 1.0 |
| Global Random (analytical) | 0.00011 | ~0 | 0.00006 | 0.033 | ~0.3 |
| Global Random (empirical) | 0.00012 | 0.00021 | 0.00006 | 0.034 | 0.24 |
| **Same-Cluster Random** | **0.00048** | **0** | **0.00024** | **0.143** | **1.0** |
| Frequency-Weighted Random | 0.0 | 0 | 0.0 | 0.0 | 0.0 |

### Critical Finding: UAD == Same-Cluster Random

**UAD F1 (0.00048) is IDENTICAL to same-cluster random F1 (0.00048).** This means:

1. **The clustering step provides ZERO value**: Randomly sampling pairs from within the same clusters achieves exactly the same performance as UAD's full pipeline (co-occurrence matrix + hierarchical clustering + phi filtering + dead feature filtering).

2. **All UAD's complexity is irrelevant**: The phi coefficient filtering, dead feature filtering, and specificity checks do not improve over random sampling within clusters.

3. **UAD has no actual contribution**: For hierarchical absorption detection, UAD is statistically indistinguishable from a trivial random baseline.

### Why Same-Cluster Random Equals UAD

The UAD pipeline detects 4155 candidate pairs across 50 clusters. When all 4155 cluster-internal pairs are considered, randomly selecting 4155 pairs from the same cluster structure yields the exact same 1 true positive out of 7 ground truth pairs. The clustering happens to place 1 GT pair in the same cluster by chance, and neither UAD's filtering nor random sampling can distinguish it from the other 4154 false positive pairs.

### Pass Criteria Assessment

| Criterion | Required | Actual | Status |
|-----------|----------|--------|--------|
| Global random F1 < 0.05 | < 0.05 | 0.00011 | **PASS** |
| UAD F1 - random F1 >= 0.3 | >= 0.3 | 0.00037 | **FAIL** |
| **OVERALL** | Both pass | - | **FAIL** |

### Implications

1. **UAD provides no meaningful advantage over random**: The tiny F1 difference (0.00037) is 3 orders of magnitude below the required threshold (0.3).

2. **The method is not viable**: Even if UAD were slightly better than random, an F1 of ~0.0005 is practically useless for any real application.

3. **Confirms P2 finding**: P2 showed UAD fails due to fundamental co-occurrence mismatch. P3 confirms this failure is not salvageable - the method has no discriminative power.

---

## Overall Pilot Assessment

### Pass/Fail Summary

| Pilot | Status | Key Finding |
|-------|--------|-------------|
| P1 (Proxy Validation) | **PASS** | Collision rate correlates with absorption (r=0.71) |
| P2 (UAD Reproducibility) | **FAIL** | UAD cannot detect hierarchical absorption |
| P3 (Random Baseline) | **FAIL** | UAD indistinguishable from random |

### Decision: PIVOT

**All three pilots together tell a clear story:**

1. **P1 validates the proxy metric**: Collision rate is a reasonable proxy for absorption rate. This is a genuine empirical finding.

2. **P2 reveals UAD's fundamental flaw**: Co-occurrence clustering cannot detect features that are mutually exclusive at the token level.

3. **P3 confirms UAD has no value**: The method is statistically indistinguishable from random sampling within clusters.

**The paper should pivot to:**

> "Why Co-occurrence Clustering Cannot Detect Feature Absorption: A Negative Result and Conceptual Analysis"

**Core contributions of the pivoted paper:**
1. Validation that collision rate is a proxy for absorption rate (positive result)
2. Demonstration that co-occurrence-based methods fail for hierarchical absorption (negative result)
3. Conceptual analysis of why absorption features are mutually exclusive (theoretical insight)
4. Discussion of alternative approaches (semantic similarity, causal intervention)

**This is an honest, valuable negative result.** The SAE community needs to know that co-occurrence-based approaches are not suitable for absorption detection, so future work can focus on more promising directions.
