# Supervisor Review: Research Contribution Assessment

**Iteration**: 2
**Date**: 2026-04-27
**Score**: 6.0 (Borderline Reject)
**Verdict**: Revise

---

## Executive Summary

The paper's central finding (H3 reversed: high-absorption features are more steerable, Spearman r=+0.35, p<0.001) is a genuinely interesting empirical result that would be of interest to the interpretability community. The full-scale H3 experiment (N=100 features) is properly designed and the steering protocol is clearly described. The honest reporting of negative results (H2 falsified, H4 regression) is commendable.

However, THREE CRITICAL FLAWS prevent acceptance:

1. **H2 metric contradiction**: Pilot results use Chanin first-letter probe (vanilla=0.225, TopK=0.066) while full-scale uses Gini absorption (vanilla=0.015, TopK=0.068). The paper presents these without distinguishing which metric produced which numbers.

2. **H5 threshold fabrication**: The paper invents a 5% threshold. The actual pass criterion was >8% (computed from pilot). Achieved: 2.51%. This is a 5.49pp shortfall, not a 'marginal miss.'

3. **Null control inconsistency**: Null controls use alpha=5 only while main H3 aggregates [1,3,5,10,20]. At alpha=5, high and low absorption are statistically identical (p=0.94). The paper claims the effect is 'driven by high-alpha conditions' - a post-hoc explanation for an internal contradiction.

---

## Dimension Scores

| Dimension | Score | Assessment |
|-----------|-------|------------|
| Novelty | 7 | The H3 reversal is genuinely novel. Prior work assumed absorption degrades steering; this shows the opposite. |
| Soundness | 5 | Core finding is sound but entanglement mechanism is stated as established fact without empirical support. |
| Experiments | 5 | Full H3 is solid (N=100). H2 has metric confusion. H4 pilot-only. H5 failed but framed as success. |
| Reproducibility | 5 | Pilot/full mixing makes reproduction ambiguous. Single-model scope limits generalizability. |

---

## Critical Issues

### 1. H2 Pilot vs Full-Scale Metric Confusion (CRITICAL)

**Evidence**: From `h2_mitigation_benchmark.json` (full-scale):
- Vanilla SAE: Gini absorption = 0.015, MSE = 1.48
- TopK SAE: Gini absorption = 0.068, MSE = 108 (4.5x INCREASE)
- JumpReLU/OrtSAE/ATM: absorption ~0.63

From `pilot_h2_summary.md` (pilot):
- Vanilla SAE: Chanin absorption = 0.225, MSE = 13.53
- TopK SAE: Chanin absorption = 0.066, MSE = 110 (70.9% REDUCTION)

The pilot uses the Chanin first-letter probe; the full-scale uses Gini absorption. These are fundamentally different metrics.

**Impact**: The paper Section 4.4 presents '70.9% absorption reduction at 8x reconstruction cost' without noting that this uses a different metric than the full-scale results (where TopK INCREASES absorption 4.5x).

### 2. H5 Threshold Fabrication (CRITICAL)

**Evidence**: From `full_h5_downstream_tasks.json` and the experimental design in `proposal.md`, the pass criterion was >8% causal delta.

From the paper Section 4.5:
- Simple task: 7.45% delta (Low - High)
- Causal task: 2.51% delta
- Paper framing: 'This marginal degradation (4.95% vs 5% threshold) is not statistically significant'

The 5% threshold does not exist in the experimental design. The actual shortfall is 5.49pp below the required >8%.

### 3. Null Control Protocol Mismatch (CRITICAL)

**Evidence**: From `pilot_h3_null.json`:
- High-absorption mean: 0.7485
- Low-absorption mean: 0.7543
- Difference: p = 0.94 (statistically identical)

The main H3 experiment aggregates across alpha=[1,3,5,10,20] and finds 18.4% difference. At alpha=5, there is no difference.

The paper explains: 'the absorption-sensitivity relationship is driven primarily by the alpha=10 and alpha=20 conditions.' This explanation was not pre-registered.

---

## Major Issues

### 4. Entanglement Hypothesis Unvalidated

The 'entanglement hypothesis' is presented as a named contribution without empirical support beyond the H3 Spearman r=+0.35. A correlation does not establish a causal mechanism. The 'hub feature' analogy is introduced without any network analysis. Section 5.1 states 'absorbed features are deeply integrated into model computation' - this is stated as established fact, not a hypothesis.

**Fix**: Present as speculative interpretation, not established contribution.

### 5. H1 Layer-wise Instability

Three experimental runs produced contradictory orderings:
- Pilot run 1: layer 8 > layer 4 (+10.6%)
- Pilot run 2: layer 4 > layer 8 (-22.9%)
- Full H1 GPT-2 atlas: layer 2 highest, monotonic DECREASE to layer 10

The paper should not claim any layer-wise pattern is established.

### 6. Full H4 Not Full-Scale

Full H4 was run in PILOT mode (3 pairs only). Spearman r declined from pilot (0.8147) to full-pilot (0.587-0.706). The paper presents this as 'full' results without acknowledging the discrepancy.

### 7. Effect Size Inconsistency

Section 4.2 says '~15% higher' but 0.1035/0.0874 = 18.4%. The abstract correctly says '18%'.

### 8. Scope Mismatch

Title and abstract promise multi-model scope ('Empirical Study' implies breadth). Gemma-2B experiments were skipped. The paper covers only GPT-2 Small layer 8.

---

## What Works

1. **H3 finding is genuinely surprising and important**: The positive correlation between absorption and steering sensitivity is counter-intuitive and worth reporting.

2. **H4 (UAS validation) is solid**: r=0.65-0.79 across layers with appropriate statistical methods.

3. **Honest negative results**: The paper correctly reports H2 as falsified and acknowledges H4 regression.

4. **Limitations section**: Acknowledges single-model scope and the alpha-dependence of the absorption-sensitivity relationship.

---

## What Would Raise the Score

| Action | Score Impact |
|--------|-------------|
| Fix H2 metric separation | +0.5 |
| Correct H5 framing (report as failed) | +0.5 |
| Align null control protocol with main experiment | +0.25 |
| Add mechanism validation for entanglement hypothesis | +0.25 |
| Replicate on Gemma-2B or explicitly scope to GPT-2 | +0.25 |

**Target**: 7.5-8 with all fixes applied.

---

## Recommendation

**Revise and resubmit.** The core finding is publishable, but the paper needs:

1. Clean separation of pilot and full-scale results with consistent metrics
2. Honest H5 reporting (failed, not marginal miss)
3. Null controls that match the main experiment protocol
4. Mechanism presented as speculation, not established fact

With these corrections, the paper would be a strong 7-7.5 submission.

---

*Review generated by Sibyl Supervisor Agent*
