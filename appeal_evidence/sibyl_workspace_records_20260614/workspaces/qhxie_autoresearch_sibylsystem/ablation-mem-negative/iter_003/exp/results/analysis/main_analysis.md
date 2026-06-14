# Iteration 3: Main Results Summary

> Date: 2026-04-29 01:41
> Analysis: sibyl-experimenter

## Overall Assessment

**UAD Status: FAILED**
**Proxy Metric Status: VALIDATED**
**Recommendation: PIVOT to negative result paper**

---

## Key Findings

### 1. Collision Rate Proxy: VALIDATED (Positive Result)

| Experiment | N Pairs | Spearman r | 95% CI | Status |
|-----------|---------|-----------|--------|--------|
| P1 (First Letters) | 10 | 0.711 | [0.219, 0.887] | PASS |
| F4 (Extended) | 56 | 0.869 | [0.780, 0.938] | PASS |

**Conclusion**: Collision rate (top-k feature overlap) is a valid proxy for true absorption rate across multiple hierarchy types.

### 2. UAD Method: FAILED (Negative Result)

| Experiment | F1 | Precision | Recall | TP | FP | Status |
|-----------|-----|-----------|--------|----|----|--------|
| P2 (Reproduction) | 0.0000 | 0.0000 | 0.1430 | 1 | 4154 | FAIL |
| P3 (vs Random) | 0.0005 | - | - | - | - | FAIL |
| F2 (Ablations) | 0.0005 | 0.0002 | 0.1429 | 1 | 4154 | FAIL |

**Key insight**: UAD F1 (0.000481) equals same-cluster random baseline (0.000481). The clustering step provides zero value.

### 3. Best Ablation Variant: K-means

- F1: 0.0037
- Precision: 0.0019
- Recall: 0.8571
- TP: 6/7

Even the best variant has near-zero precision due to massive false positives.

---

## Root Cause

**Primary**: Absorption features are mutually exclusive at the token level.

Features that absorb the same parent concept fire on DIFFERENT tokens representing different child concepts. They never activate on the same token.

**Why UAD fails**: UAD uses co-occurrence clustering (phi coefficient) to find features that fire TOGETHER. But absorption features fire on mutually exclusive instances, so their co-occurrence is near zero.

**Why proxy works**: Collision rate measures structural similarity of feature responses, not co-occurrence. Two child concepts may share the same absorbing feature in their top-k even though they never appear together.

---

## Paper Positioning

**Title**: "Why Co-occurrence Clustering Cannot Detect Feature Absorption in Sparse Autoencoders"

**Type**: Negative result with constructive insight

**Contributions**:
1. Empirical demonstration that UAD fails (F1=0.0005)
2. Validation that collision rate IS valid (r=0.87)
3. Root cause identification: token-level mutual exclusivity
4. Proposed alternatives: decoder weight similarity, causal intervention

---

## Honest Limitations

- Only tested on GPT-2 Small with gpt2-small-res-jb SAE
- Only 7 ground truth absorption pairs
- Collision rate tested on 56 pairs across 2 hierarchy types
- No causal validation of proposed alternatives

---

## Decision

**PIVOT**: Abandon UAD as a detection method. Write negative result paper focusing on:
1. Why co-occurrence clustering is the wrong tool for absorption detection
2. Validation of collision rate as a proxy metric
3. Proposed alternative approaches for future work
