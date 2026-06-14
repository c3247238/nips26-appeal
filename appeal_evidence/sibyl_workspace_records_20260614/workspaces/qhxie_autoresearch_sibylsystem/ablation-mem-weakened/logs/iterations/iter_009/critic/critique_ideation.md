# Critique of Ideation: Feature Absorption as Optimal Compression

## Overview

The research question (is absorption a failure mode or benign artifact?) is well-posed. The optimal compression framing (cand_g) is intellectually coherent. However, the ideation has critical flaws in hypothesis construction and statistical rigor.

## Critical Issues

### 1. H9 Co-occurrence Hypothesis is a Tautology (CRITICAL)

**Problem**: H9 operationalizes absorption as p_11 + absorption_rate = 1.0, which is definitional. The "correlation" of r=-1.0 is a mathematical identity, not an empirical finding.

**Timeline**:
- Iteration 1: H9 was identified as tautological
- Iteration 5: H9 confirmed as TAUTOLOGICAL in the proposal
- Still referenced in Section 4.8 Table 5 as if it were a valid empirical finding

**Verdict**: H9 must be removed from the paper entirely. It adds no scientific value.

### 2. H1b is Post-Hoc Cherry-Pick from H1 (CRITICAL)

**Problem**: H1b (delta-corrected steering) was not pre-registered as the primary hypothesis. It emerged post-hoc after observing that raw steering showed no correlation. Running multiple analyses and selecting the one that shows p<0.05 is p-hacking.

**Timeline**:
- H1 (pre-registered): raw steering correlation with absorption
- Observed: zero correlation (r=+0.008 L4, r=-0.301 L8, p>0.05)
- H1b (post-hoc): delta-corrected steering
- Observed: r=-0.431, p=0.028 (uncorrected)
- Correction applied: Bonferroni p=0.334, BH-FDR q=0.107

**Verdict**: H1b cannot be presented as evidence of a real effect. Either pre-register H1b before data collection, or honestly report it as an exploratory analysis that did not survive correction.

### 3. H6 Primary Hypothesis Construction is Flawed (CRITICAL)

**Problem**: H6 claims the local inhibition graph "predicts absorption pairs." The hypothesis was tested and decisively falsified (precision@20=0.0, Fisher p=1.0). The paper attempts to salvage this as a "valuable negative result" but the framing was wrong from the start.

**Root cause**: The hypothesis conflated the LCA-theoretical framework (valid) with graph-based prediction (not validated). The paper should have separated:
- H6a: LCA-SAE structural correspondence is exact for tied-weight SAEs (theoretical)
- H6b: Local inhibition graph predicts absorption pairs (empirical, falsified)

**Verdict**: The theoretical contribution stands independently. H6b should be reported as falsified, not as a "valuable negative result" that somehow contributes.

### 4. H10 Homeostatic Rebalancing Deferred Without Justification (MAJOR)

**Problem**: H10 was not executed at all, yet the paper discusses it extensively in Section 3.4, 5.3, and 5.5. The justification ("deferred pending improved graph construction") is circular: the graph was supposed to identify parent-child relationships, but since it failed, the repair cannot be tested.

**Verdict**: Either execute H10 or remove all discussion of homeostatic rebalancing as a practical contribution.

## What Works

1. **Research question framing**: "Is absorption a failure mode or benign artifact?" is well-posed and addresses a real gap in the literature.

2. **Random baseline comparison (H7)**: Comparing trained vs. random SAEs is a strong methodological contribution. This is the paper's most valid empirical finding.

3. **Optimal compression framing**: The rate-distortion interpretation is intellectually coherent and provides a plausible explanation for why absorption persists (it is compression-optimal, not pathological).

4. **Precision-recall asymmetry (H5)**: This is the one robust replicable finding. Precision=1.0 universally at k>=5 while recall varies (0.05-1.0). This is consistent across iterations.

## Ideation-Specific Recommendations

1. **Pre-register hypotheses before data collection** — H1b emerged post-hoc and cannot be treated as confirmatory evidence.

2. **Remove H9** — it is a tautology, not an empirical finding.

3. **Separate theoretical from empirical claims** — H6a (LCA framework) is valid; H6b (graph prediction) was falsified. Do not conflate them.

4. **Execute H10 or remove it** — deferred is not a status, it is an omission.

5. **Acknowledge power limitations upfront** — n=26 provides insufficient power for medium effects. This should be stated in the methodology, not explained away post-hoc.

6. **Consider metric validation framing** — if the graph predictions are abandoned, the paper could pivot to a metric validation study (trained vs. random SAE comparison validates the Chanin metric's sensitivity to dictionary structure).