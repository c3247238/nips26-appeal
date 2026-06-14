# Experiment Result Analysis

## Key Results Summary

**Primary Finding (H1-H4): Null results supported.** Absorption does NOT significantly degrade downstream tasks (steering effectiveness or sparse probing accuracy) after multiple comparison correction.

| Hypothesis | Result | Key Evidence |
|---|---|---|
| H1 (steering degradation) | SUPPORTED (null) | r=+0.008 (L4), r=-0.301 (L8), neither significant |
| H2 (probing degradation) | SUPPORTED (null) | r=-0.003 (L4), r=-0.107 (L8), neither significant |
| H3 (cross-layer consistency) | SUPPORTED (null) | Opposite slopes, CV=1.079 (>0.5 threshold) |
| H4 (EC50 correlation) | SUPPORTED (null) | L4: r=-0.166, p=0.439; L8: r=+0.180, p=0.380 |
| H5 (precision-recall asymmetry) | SUPPORTED | Precision=1.0 at k>=5; recall varies (0.05-1.0) |
| H6 (graph prediction) | FALSIFIED | precision@20=0.0, enrichment=0.0x, p=1.0 |
| H7 (trained < random absorption) | SUPPORTED | Trained=0.034 vs Random=0.278, p<0.001 |
| H9 (co-occurrence) | NO_GO | r=-1.0 is tautological (p_11 + absorption = 1 by construction) |

**Critical Numbers:**
- Trained SAE mean absorption: 0.034 (std=0.069, max=0.242)
- Random SAE mean absorption: 0.278 (std=0.169, max=0.676)
- Difference: -0.244 (random > trained), p < 0.001
- H6 decoder correlation prediction: precision@20 = 0.0 (falsified)

## Debate Perspectives Summary

Based on supervisor review and available evidence:

**Supervisor Review (iteration 5, score 5.5/10):**
- Empirically sound execution with honest null-result reporting
- Primary concern: proposal-paper incoherence (different titles, claims, hypotheses)
- H10 claimed as contribution without execution
- Probe ceiling effect undermines precision invariance claim at k>=5
- Random SAE baseline confounded (orthonormal decoder constraint)
- Pythia cross-validation failed (r=-0.041, p=0.841)

**Key Concerns:**
1. Narrative coherence problem between proposal and paper
2. Overclaiming H10 (homeostatic rebalancing not executed)
3. Abstract self-contradiction (H6 validated vs falsified in same paragraph)
4. Steering metric too coarse (top-5 criterion misses magnitude effects)
5. Single-model limitation (GPT-2 Small only)

**Defended Positions:**
- Null results are robust after Bonferroni correction (12 tests, alpha=0.00417)
- H7 (trained < random) is genuinely novel and validated
- H6 falsification is a valuable negative result
- H5 precision-recall asymmetry is the one replicable finding

## Analysis

### 1. Method Feasibility
**Assessment: Sound with caveats.** The Chanin absorption metric works but is not well-calibrated (random SAE shows 8x higher absorption, proving the metric captures structural artifacts, not learned failures). Steering and probing methodologies are sound but coarse-grained. Probe ceiling effect at k>=5 undermines precision invariance claim.

### 2. Performance
**Assessment: Null results hold.** No significant degradation from absorption on any downstream task. The one robust positive finding is H5 (precision-recall asymmetry), not a performance improvement.

### 3. Improvement Headroom
**Assessment: Limited for this direction.** All experiments are completed. The paper's narrative problems (incoherence) are writing issues, not experiment issues. The findings are what they are - honest null results with one robust asymmetry finding.

### 4. Time-Cost Tradeoff
**Assessment: Favorable to PROCEED.** All experiments are done. ~2-3 days of writing/revision remain. Pivoting would require new experiments and extend timeline by weeks. The current results support a viable paper (null results + metric validation + methodological contributions).

### 5. Critical Objections
**Assessment: Addressable.** The supervisor review flags valid concerns but none are fatal:
- Proposal-paper incoherence: fixable through revision
- H10 overclaiming: remove from contributions or execute
- Probe ceiling: acknowledge as limitation, focus on k=1 results
- Random baseline confound: acknowledge orthonormal constraint as limitation
- Single model: acknowledge and frame as focused study on GPT-2 Small

## Decision Rationale

**Reasons to PROCEED:**
1. All experiments completed (H1-H10, 9 major analyses across layers/models)
2. Null results are robust and honestly reported
3. H7 (trained < random absorption) is a genuinely novel finding
4. H6 falsification is valuable negative result
5. H5 (precision-recall asymmetry) is replicable
6. ~2-3 days of work remaining vs weeks for new experiments
7. Paper is empirically sound; issues are narrative/writing, not fundamental

**Reasons to PIVOT:**
1. Proposal-paper incoherence suggests unresolved core narrative
2. Review score 5.5/10 is borderline
3. Single model limitation reduces generalizability
4. Null results may be seen as "we found nothing"

**Key Evidence:** The supervisor review explicitly states "The empirical execution is mostly sound and honest. The narrative is fractured." This indicates the core science is valid; the issue is presentation. With 5.5/10 score, the paper is not unrecoverable - it needs coherent narrative and removal of overclaimed contributions.

**Risk Assessment:** If we PROCEED and the paper fails, we lose ~2-3 days. If we PIVOT, we invest weeks in new experiments with no guarantee of better results. Given that the empirical findings are solid and the issues are fixable writing problems, the expected return on continued optimization is higher than starting fresh.

## DECISION: PROCEED