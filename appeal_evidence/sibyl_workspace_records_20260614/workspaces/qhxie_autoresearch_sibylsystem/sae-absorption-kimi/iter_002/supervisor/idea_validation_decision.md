# Idea Validation Decision

## Pilot Evidence Summary

The pilot (`pilot_semantic_probe`) on 2 SAEs (MatryoshkaBatchTopK and TopK) with 3 WordNet hierarchies and 2 control pairs completed successfully and met all pass criteria:
- Numerically stable absorption scores
- All probe AUROCs = 1.0 (> 0.6 threshold)
- Expected ordering: Matryoshka (0.283) < TopK (0.339)
- Hierarchy > control for at least 1 SAE: True

However, the **full experimental suite** revealed critical anomalies that undermine confidence in the current implementation:

1. **Random-SAE Red Flag**: The Random SAE's semantic-hierarchy absorption (0.352) is *identical* to the Standard SAE's score (0.352) across all 10 hierarchies. This is statistically implausible and strongly suggests a data-handling bug or unintended sharing of results between the two configurations.

2. **H2 Reversed**: The paired t-test shows semantic-hierarchy absorption (mean = 0.235) is *significantly lower* than non-hierarchy control absorption (mean = 0.331), t = -4.748, p = 0.0032. This is the opposite of the hypothesized direction and raises serious questions about whether the custom pipeline is measuring the intended construct.

3. **H1 Inconclusive**: Pearson r between first-letter and semantic-hierarchy absorption = 0.463 (95% CI [-0.389, 0.981]), failing to reach the > 0.6 threshold and spanning zero.

4. **GPT-2 Replication Divergence**: GPT-2 small showed near-zero hierarchy absorption (Standard = 0.0, TopK = 0.003) versus substantial values on Pythia-160M. This model-specific divergence is unexpected and warrants methodological scrutiny.

## Decision Matrix

### Candidate A: Construct Validity of SAEBench Absorption Metric

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 3 | Pilot passed, but full results show severe anomalies (Random = Standard, H2 reversed) |
| Hypothesis survival | 0.25 | 2 | H1 inconclusive, H2 rejected (reversed), H3 inconclusive |
| Path to full result | 0.20 | 2 | Major implementation red flags; cannot trust current pipeline without debugging |
| Novelty | 0.15 | 4 | First systematic construct-validity study of the dominant absorption metric |
| Resource efficiency | 0.10 | 2 | ~1 GPU-hour already spent; continuing on a buggy pipeline would be wasteful |

**Weighted Score**: 0.30×3 + 0.25×2 + 0.20×2 + 0.15×4 + 0.10×2 = **2.60**

### Candidate B: FastProbe-Absorb

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 3 | No pilot run; feasibility inferred from existing infrastructure |
| Hypothesis survival | 0.25 | 3 | Hypotheses are plausible but untested |
| Path to full result | 0.20 | 3 | Could leverage SAEBench codebase, but scope is moderate |
| Novelty | 0.15 | 3 | Useful tool, but less novel than a construct-validity study |
| Resource efficiency | 0.10 | 4 | Likely 30-45 min pilot; well within budget |

**Weighted Score**: 0.30×3 + 0.25×3 + 0.20×3 + 0.15×3 + 0.10×4 = **3.10**

### Candidate C: Rate-Distortion Origin (Theory)

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 2 | No empirical pilot; theoretical only |
| Hypothesis survival | 0.25 | 3 | Structurally plausible but no data yet |
| Path to full result | 0.20 | 2 | Proving a general theorem is high-risk within current constraints |
| Novelty | 0.15 | 4 | Strong standalone theory contribution |
| Resource efficiency | 0.10 | 3 | Low GPU cost, but high intellectual risk |

**Weighted Score**: 0.30×2 + 0.25×3 + 0.20×2 + 0.15×4 + 0.10×3 = **2.55**

### Candidate D: Validity-Aware Analysis (Contrarian)

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 2 | No pilot run; would require causal patching infrastructure |
| Hypothesis survival | 0.25 | 2 | Ambitious and potentially confounded |
| Path to full result | 0.20 | 2 | 45 min pilot + activation patching is nontrivial |
| Novelty | 0.15 | 3 | Challenging assumptions is valuable but not unprecedented |
| Resource efficiency | 0.10 | 2 | Higher resource cost for uncertain payoff |

**Weighted Score**: 0.30×2 + 0.25×2 + 0.20×2 + 0.15×3 + 0.10×2 = **2.15**

## Decision Rationale

No candidate scores ≥ 3.5, so **ADVANCE is ruled out**. The front-runner (Candidate A) scores 2.60, placing it squarely in the **REFINE** zone. However, the rationale for refinement is not "minor methodological tweaks" — it is **fundamental pipeline debugging**.

The specific evidence triggering REFINE rather than PIVOT:
- The pilot *did* produce numerically stable, correctly ordered results.
- The research question remains important and novel.
- The anomalies in the full experiment are severe enough that they likely stem from **implementation bugs rather than a fundamentally flawed idea**. A Random SAE cannot plausibly produce *exactly* the same scores as a Standard SAE on 10 hierarchies unless there is a data-handling error.

If the bugs are fixed and the results still show no correlation or reversed hierarchy specificity, then a **PIVOT** would be warranted in the next validation round.

## Next Actions

1. **Debug the custom absorption pipeline**: Investigate why Random SAE and Standard SAE produce identical semantic-hierarchy and non-hierarchy control scores. Check for cached results, shared file paths, or incorrect SAE loading logic.
2. **Re-run the full semantic-hierarchy and non-hierarchy experiments** on the corrected pipeline for all 8 SAEs.
3. **Re-run the GPT-2 replication** with the fixed code to verify cross-model consistency.
4. **Re-compute statistical analysis** only after the above anomalies are resolved.
5. If the fixed pipeline still fails H1/H2, pivot to **Candidate B (FastProbe-Absorb)** as the backup direction.

SELECTED_CANDIDATE: cand_a
CONFIDENCE: 0.40
DECISION: REFINE
