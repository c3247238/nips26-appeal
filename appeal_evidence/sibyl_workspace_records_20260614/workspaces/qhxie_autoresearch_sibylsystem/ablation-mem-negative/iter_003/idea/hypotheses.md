# Research Hypotheses: Iteration 3 — Negative Result on Co-occurrence Clustering for Absorption Detection

## Primary Hypotheses

### H1: Co-occurrence Clustering Fails for Absorption Detection
**Statement**: The UAD co-occurrence clustering method achieves F1 <= 0.01 for detecting hierarchical feature absorption, statistically indistinguishable from random sampling within clusters.

**Status**: CONFIRMED (F1 = 0.00048, same-cluster random F1 = 0.00048)

**Evidence**:
- Full UAD pipeline: F1 = 0.00048, Precision = 0.024%, Recall = 14.3%
- Same-cluster random baseline: F1 = 0.00048 (identical)
- All ablation variants fail similarly

**Falsification Criteria** (pre-registered):
- If F1 > 0.01, H1 would be falsified
- If UAD F1 > same-cluster random F1 by > 0.01, H1 would be falsified

---

### H2: Token-Level Mutual Exclusivity Is the Root Cause
**Statement**: Absorption features are mutually exclusive at the token level---they fire on different tokens representing different child concepts---making co-occurrence-based detection inherently unsuitable.

**Status**: CONFIRMED

**Evidence**:
- Feature 11513 fires ONLY on "three" (activation = 29.4)
- Feature 24189 fires on "four" through "eight" (activations 14.9-18.9)
- Feature 11513 and 24189 NEVER co-occur on any token
- Phi coefficients between absorption features are near-zero or negative

**Falsification Criteria**:
- If any two absorption features co-occur on > 5% of tokens, H2 would be falsified
- If absorption features show positive phi correlation, H2 would be falsified

---

### H3: Collision Rate Is a Robust Proxy for Absorption Rate
**Statement**: Collision rate (Jaccard overlap of top-K features per concept) correlates with true absorption rate at Spearman r >= 0.5 across diverse concept hierarchies.

**Status**: CONFIRMED (r = 0.869, n = 56, 95% CI [0.780, 0.938])

**Evidence**:
- Pilot (first letters): r = 0.711, n = 10, CI [0.219, 0.887]
- Full (numbers + punctuation): r = 0.869, n = 56, CI [0.780, 0.938]
- Numbers only: r = 0.598, n = 28
- Punctuation only: r = 0.693, n = 28
- Bootstrap CI excludes zero in all cases

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
- No dead filter: F1 = 0.00048 (identical)
- No phi filter: F1 = 0.00048 (identical)
- No clustering: F1 = 0.000056 (worse)
- Single linkage: F1 = 0.0 (worse)
- K-means: F1 = 0.0037 (best but still near-zero)

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

## Hypothesis Dependency Graph

```
H1 (co-occurrence clustering fails)
  |
  +---> H2 (mutual exclusivity is root cause) [explains H1]
  |
  +---> H-S1 (ablations don't help) [confirms H1 is structural]

H3 (collision rate proxy) [independent validation]
  |
  +---> H-S2 (decoder weight similarity) [future direction enabled by H3]
```

## Pre-registered Analysis Plan

1. **Primary Analysis**: Report F1 / Precision / Recall with exact counts (not just rates)
2. **Baseline Comparison**: Compare to same-cluster random, global random, and frequency-weighted random
3. **Root Cause Analysis**: Token-level activation heatmaps for all identified absorption features
4. **Proxy Validation**: Spearman correlation with bootstrap 95% CI across hierarchy types
5. **Negative Result Handling**: Report all falsified hypotheses prominently with exact observed values

## Go/No-Go Gates (Retrospective)

| Gate | Condition | Outcome |
|------|-----------|---------|
| G1 | H1: F1 <= 0.01 | **PASSED** (F1 = 0.00048) |
| G2 | H3: r >= 0.5 | **PASSED** (r = 0.869) |
| G3 | H2: Mutual exclusivity confirmed | **PASSED** (token-level evidence) |

**Decision**: All gates passed in the negative direction. Proceed with negative result paper.
