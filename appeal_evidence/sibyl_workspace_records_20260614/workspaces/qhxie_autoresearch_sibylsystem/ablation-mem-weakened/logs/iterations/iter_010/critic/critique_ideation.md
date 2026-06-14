# Critique: Ideation (Iteration 9/10)

## Overview

The ideation process converged on cand_g ("Feature Absorption as Optimal Compression") after 6 perspectives and 9 iterations. The convergence is documented but the front-runner has a fundamental problem: its primary empirical prediction (H6: local inhibition graph predicts absorption pairs) is falsified, while the null result and metric validation findings that actually advance the paper are treated as secondary.

## Critical Issues

### 1. Hypothesis H6 is Falsified but Remains Central (CRITICAL)

**Problem**: H6 (precision@20=0.0) is the primary predictive hypothesis for the cand_g front-runner. The LCA-SAE correspondence is presented as the intellectual backbone, but it predicts that graph edges should correspond to absorption pairs—and they don't.

**What the evidence says**:
- H6: precision@20=0.0, p=1.0 (Fisher exact) - FALSIFIED
- H8: r=+0.12, p=0.55 (graph statistics don't predict at-risk features) - NOT SUPPORTED

**What should have happened**: When H6 falsified, the ideation should have pivoted to Alternative A (Metric Validation Study) or reframed around H7 (trained < random absorption). Instead, the paper kept the LCA framing and treated H6 falsification as a "nuance" rather than the central finding.

**Recommendation**: The idea should be reframed as "Metric Validation + Null Results" not "Optimal Compression". The LCA connection can remain as mechanistic background, but it cannot be the primary contribution when its main prediction fails.

### 2. Metric Validation Finding is Underweighted (CRITICAL)

**Problem**: The strongest empirical finding is H10 (random SAE shows 8x higher absorption: 0.278 vs 0.034, p<0.001). This reveals that the Chanin absorption metric is sensitive to dictionary structure, not specifically to learned pathology. This is worth an entire paper, but it's treated as a "methodological contribution" and buried in the middle.

**Evidence from pilot_summary.json**:
```json
"h10_random_sae_baseline": {
  "status": "completed",
  "go_no_go": "GO",
  "confidence": 0.95,
  "key_finding": "Random SAE shows 8x higher absorption than trained SAE (0.278 vs 0.034), opposite to prediction"
}
```

**Recommendation**: Consider promoting Alternative A ("Feature Absorption Metrics Measure Dictionary Structure, Not Learned Pathology") as the primary front-runner. This is a clean, falsifiable, and genuinely novel contribution.

### 3. Multiple Candidates Failed but No Pivot (MAJOR)

**What failed**:
- H6: Graph predicts absorption pairs - FALSIFIED
- H8: Graph statistics predict at-risk features - NOT SUPPORTED
- H9: Co-occurrence strength predicts absorption - TAUTOLOGICAL
- H1b: Delta-corrected steering correlation - NOT SIGNIFICANT after MCP

**What succeeded**:
- H1-H4: Null results (absorption doesn't degrade downstream tasks) - SUPPORTED
- H5/H7: Precision-recall asymmetry (precision=1.0, recall varies) - SUPPORTED
- H10: Random SAE has higher absorption than trained - SUPPORTED

**The problem**: The ideation process identified cand_g as "optimal compression" based on the assumption that H6 would succeed. When it didn't, no pivot occurred. The paper proceeds with a framework whose main prediction failed.

**Recommendation**: Document the pivot conditions explicitly and trigger one:
- If H6 falsified → pivot to cand_h (pure null-result + methodology) or Alternative A (metric validation)
- The current paper tries to have it both ways: claims the LCA framework is the contribution while acknowledging H6 failed

### 4. Novelty Claim is Weak (MAJOR)

**Problem**: The proposal says "First random SAE baseline comparison for absorption metrics specifically (vs. general metric comparison in Sanity Checks)". But:
1. The LCA-SAE structural correspondence (G=W_dec^T W_dec) is mathematically immediate from the definitions—it's not a deep insight
2. Korznikov et al. (2026) "Sanity Checks" already did frozen/random baselines for SAE metrics

**Assessment from proposal.md**:
> | Korznikov et al. (2026) - "Sanity Checks" | Frozen/random baselines match trained on standard metrics | Partial overlap - general metrics vs. absorption specifically |

But the paper doesn't clearly articulate what's new. The random baseline comparison for absorption metrics specifically is the differentiator, but the paper doesn't lean into this enough.

**Recommendation**: Strengthen the novelty claim by making the metric validation the primary contribution, not the LCA connection.

## Summary

The ideation process was thorough (6 perspectives, 9 iterations) but got stuck on cand_g when its primary hypothesis failed. The strongest findings (null results + metric validation) are treated as secondary to a framework whose main prediction failed. A pivot to Alternative A or cand_h would produce a stronger paper.

**Action items**:
1. Reconsider whether cand_g is the right front-runner given H6 falsification
2. Promote H10 (metric validation) to primary contribution
3. Consider Alternative A (Metric Validation Study) as primary
4. Clarify what's actually novel: the random baseline comparison for absorption metrics specifically