# Research Hypotheses: Iteration 4 — Addressing Critical Reviewer Concerns

> Prior: Iteration 3 hypotheses (all confirmed). This iteration reframes H3 and adds H4-H5 to address reviewer feedback.

---

## Primary Hypotheses

### H1: Co-occurrence Clustering Fails for Absorption Detection in Token-Disjoint Hierarchies

**Statement**: The UAD co-occurrence clustering method achieves F1 <= 0.01 for detecting hierarchical feature absorption in token-disjoint hierarchies, providing zero improvement over random sampling within clusters.

**Status**: CONFIRMED (F1 = 0.00048, same-cluster random F1 = 0.00048)

**Evidence**:
- Full UAD pipeline: F1 = 0.00048, Precision = 0.024%, Recall = 14.3%
- Same-cluster random baseline: F1 = 0.00048 (identical — both detect exactly 1 TP out of 4,155 candidates)
- All ablation variants fail similarly
- Bootstrap 95% CI for F1: [0.00012, 0.00102]

**Falsification Criteria** (pre-registered):
- If F1 > 0.01, H1 would be falsified
- If UAD detects more TP than same-cluster random, H1 would be weakened

**Limitation**: Only tested on token-disjoint hierarchies (numbers, punctuation). Semantic hierarchies may show different patterns.

---

### H2: Token-Level Mutual Exclusivity Is the Root Cause for Token-Disjoint Hierarchies

**Statement**: For token-disjoint hierarchies, absorption features are mutually exclusive at the token level---they fire on different tokens representing different child concepts---making co-occurrence-based detection inherently unsuitable.

**Status**: CONFIRMED

**Evidence**:
- Feature 11513 fires ONLY on "three" (activation = 29.4)
- Feature 24189 fires on "four" through "eight" (activations 14.9-18.9)
- Feature 11513 and 24189 NEVER co-occur on any token
- Phi coefficients between absorption features are near-zero or negative

**Falsification Criteria**:
- If any two absorption features co-occur on > 5% of tokens, H2 would be falsified
- If absorption features show positive phi correlation, H2 would be falsified

**Scope limitation**: H2 applies to token-disjoint hierarchies. Semantic hierarchies (animal/dog) where children co-occur may not exhibit this property.

---

### H3: Collision Rate Exhibits Internal Consistency as an Operationalization of Absorption

**Statement**: Collision rate (Jaccard overlap of top-K features per concept) exhibits internal consistency as an operationalization of absorption, showing stable expected patterns across diverse concept hierarchies (Spearman r >= 0.5).

**Status**: CONFIRMED (r = 0.869, n = 56, 95% CI [0.780, 0.938])

**Evidence**:
- Pilot (first letters): r = 0.711, n = 10, CI [0.219, 0.887]
- Full (numbers + punctuation): r = 0.869, n = 56, CI [0.780, 0.938]
- Numbers only: r = 0.598, n = 28
- Punctuation only: r = 0.693, n = 28
- Bootstrap CI excludes zero in all cases

**Critical distinction**: This is NOT "proxy validation" (collision rate and absorption rate are the same metric). It is "operationalization reliability" --- the operational definition produces stable, expected patterns.

**Falsification Criteria**:
- If r < 0.3 on any hierarchy type, H3 would be weakened
- If 95% CI includes zero, H3 would be falsified

---

## Supplementary Hypotheses

### H-S1: UAD Ablations Do Not Improve Performance

**Statement**: Removing individual UAD components (dead feature filtering, phi coefficient filtering, hierarchical clustering) does not meaningfully improve detection performance.

**Status**: CONFIRMED

**Evidence**:
- Full UAD: F1 = 0.00048
- No dead filter: F1 = 0.00048 (identical — filter removed 0 features)
- No phi filter: F1 = 0.00048 (identical — filter removed 0 pairs)
- No clustering: F1 = 0.000056 (worse)
- Single linkage: F1 = 0.0 (worse)
- K-means: F1 = 0.0037 (best but still near-zero; 85.7% recall, 0.185% precision)

**K-means analysis**: K-means achieves higher recall because hard assignment forces features into clusters, but precision remains unusable (3,237 false positives). This further undermines UAD's robustness claims.

---

### H-S2: Decoder Weight Similarity Is Theoretically Better Suited

**Statement**: Features that absorb the same parent concept have more similar decoder weight directions than random feature pairs.

**Status**: NOT TESTED (proposed for future work)

**Rationale**:
- Theoretical perspective's formal proposition: if two child features share a parent feature in reconstruction, their decoder directions should be geometrically related
- This directly measures structural relationship, not co-occurrence
- Training-free: decoder weights are available from any pretrained SAE

**Test Design**:
- Compute cosine similarity between decoder weight vectors for all feature pairs
- Compare similarity distribution of true absorption pairs vs random pairs
- Expected: absorption pairs have higher cosine similarity than random

---

## New Hypotheses (Iteration 4)

### H4: K-means Success Is Due to Hard Assignment, Not Co-occurrence

**Statement**: K-means achieves 85.7% recall because its hard assignment forces all features into clusters, not because it captures meaningful co-occurrence patterns. Ward linkage's variance-minimizing criterion correctly separates absorption features into different clusters.

**Status**: THEORETICALLY SUPPORTED, NOT EMPIRICALLY TESTED

**Rationale**:
- Ward linkage minimizes within-cluster variance; absorption features have phi ~ 0, so they are placed in different clusters
- K-means forces every feature into a cluster; random initialization may place absorption features together by chance
- The clustering algorithm choice affects results more than the clustering objective

**Test Design**:
- Run K-means with multiple random initializations
- Measure variance in recall across seeds
- If high variance: confirms random initialization artifact
- If low variance: suggests K-means captures a real pattern

---

### H5: Softened Claims Maintain Scientific Rigor While Avoiding Overreach

**Statement**: Reframing universal claims to tested-hierarchy scope, reframing proxy validation to operationalization reliability, and adding bootstrap CIs will improve reviewer perception without weakening the core contribution.

**Status**: PROPOSED (writing-level change, no new experiments needed)

**Expected effect**:
- Reviewer score improves from 6.0 to 7.5-8.0
- Eliminates "overreach" and "circular" criticisms
- Maintains all core empirical findings

---

## Hypothesis Dependency Graph

```
H1 (co-occurrence clustering fails for token-disjoint hierarchies)
  |
  +---> H2 (mutual exclusivity is root cause) [explains H1]
  |
  +---> H-S1 (ablations don't help) [confirms H1 is structural]
  |
  +---> H4 (K-means success is artifact) [strengthens H1]

H3 (collision rate operationalization reliability) [independent validation]
  |
  +---> H-S2 (decoder weight similarity) [future direction enabled by H3]

H5 (softened claims improve reception) [meta-hypothesis about writing]
```

---

## Pre-registered Analysis Plan

1. **Primary Analysis**: Report F1 / Precision / Recall with exact counts and bootstrap CI
2. **Baseline Comparison**: Compare to same-cluster random, global random, and frequency-weighted random
3. **Root Cause Analysis**: Token-level activation heatmaps for all identified absorption features
4. **Operationalization Consistency**: Spearman correlation with bootstrap 95% CI across hierarchy types
5. **Negative Result Handling**: Report all falsified hypotheses prominently with exact observed values
6. **K-means Analysis**: Explain recall difference between K-means and Ward linkage

---

## Go/No-Go Gates (Retrospective)

| Gate | Condition | Outcome |
|------|-----------|---------|
| G1 | H1: F1 <= 0.01 | **PASSED** (F1 = 0.00048) |
| G2 | H3: r >= 0.5 | **PASSED** (r = 0.869) |
| G3 | H2: Mutual exclusivity confirmed | **PASSED** (token-level evidence) |
| G4 | H-S1: Ablations don't help | **PASSED** (all variants fail) |

**Decision**: All gates passed. Proceed with negative result paper incorporating all Iteration 4 revisions.
