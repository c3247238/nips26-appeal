# Ideation Critique: Encoder-Driven Feature Absorption

## Summary

The proposal is well-structured with clear hypotheses, but several critical issues undermine confidence in the research direction: (1) the B>D anomaly is presented as a finding when it may be a statistical artifact, (2) H_Downstream (the most important practical question) is listed but never tested, (3) the efficient coding reframing of H2 is post-hoc rationalization.

## Proposal Quality

### Strengths

1. **Clear research questions**: The 5 questions in Section 2 are well-framed and map to testable hypotheses
2. **Honest negative result reporting**: H2 falsification is documented with a mechanistic explanation (efficient coding)
3. **B>D anomaly highlighted**: The decoder regularization insight is genuinely interesting and actionable

### Weaknesses

### Critical: H_Downstream Listed But Never Tested

The proposal's most valuable contribution (contrarian's challenge: does absorption matter for downstream utility?) is listed as a hypothesis but never tested in the paper. The writing/paper.md shows no H_Downstream results.

This is a significant gap. SAEBench has already shown proxy metrics don't predict practical performance. If absorption also doesn't predict downstream failure, the entire research direction may be misallocated effort.

**Recommendation**: Either test H_Downstream or explicitly remove it from the paper's contributions and frame the paper as purely a measurement methodology contribution.

### Critical: B>D Anomaly May Be Noise

The B>D finding (0.490 vs 0.484) has:
- Difference of 0.006
- No p-value reported for B vs D comparison
- No confidence intervals

At p < 0.05 significance, a 0.006 difference across 5 seeds could easily be noise. The paper presents this as a "central finding" without statistical validation.

**Recommendation**: Add explicit t-test or confidence interval for B vs D comparison. If not significant, don't call it an anomaly.

### Major: H2 Efficient Coding Reframe is Post-Hoc

H2 predicted negative correlation (rho < -0.3). Observed: rho = +0.171 (positive). The proposal reframes this as "efficient coding" after seeing the data.

This is acceptable as a post-hoc interpretation, but the framing shouldn't imply the efficient coding hypothesis was pre-registered. The original competitive exclusion hypothesis was wrong - that's a valid honest negative result without needing a post-hoc rescue.

**Recommendation**: Report H2 as a failed hypothesis. The efficient coding discussion belongs in the Discussion section, not as the primary interpretation.

### Major: H_Pareto Suspended But Still Listed

The sensitivity-absorption Pareto frontier is a central theoretical contribution but is suspended due to formula bugs. Yet the proposal still lists it as a contribution and includes it in Figure 2 planning.

**Recommendation**: Remove H_Pareto from contributions until formula is verified. Don't promise results you can't deliver.

### Minor: H_Safe Uses Placeholder Features

The proposal correctly identifies that prior pilot used arbitrary indices (1024, 2048, etc.) as safety features. However, the proposal doesn't specify WHERE to get validated Neuronpedia features or how many validated features are available.

**Recommendation**: Add explicit feature selection criteria and expected sample sizes from Neuronpedia.

## Hypothesis Table Issues

| Hypothesis | Proposal Status | Paper Status | Issue |
|------------|----------------|--------------|-------|
| H1 | PASSED | PASSED | Consistent |
| H_Mech | PASSED + anomaly | PASSED | B>D needs p-value |
| H2 | FAILED | FAILED | Consistent |
| H3 | PASSED | PASSED | Consistent |
| H_Safe | NOT TESTED | FAILED (p=0.665) | Paper ran it, proposal didn't |
| H_Comp | PASSED (pilot) | Not in paper | Missing from paper |
| H_Pareto | SUSPENDED | Not in paper | Correctly omitted |
| H_Downstream | NOT TESTED | Not in paper | Missing from both |

**Key Gap**: H_Comp (hierarchy strength dependence) is in the proposal but not the paper.

## Novelty Assessment

The proposal claims novelty for:
1. Encoder-driven mechanism - **Novel if B>D is real**
2. Sensitivity-absorption Pareto frontier - **Suspended**
3. Safety-critical feature analysis - **9/10 novel if validated features used**
4. Downstream utility test - **8/10 but never done**

The paper actually delivers:
1. Multi-child proportional ablation methodology - **Incremental over single-child**
2. Factorial decomposition showing geometric vs learned - **Novel design, B>D issue**
3. Safety analysis on validated features - **9/10 but null result**
4. Negative result on frequency correlation - **Honest but not novel contribution**

## Recommendations

1. **Either test H_Downstream or remove it from contributions**
2. **Add p-value for B>D comparison** or don't call it an anomaly
3. **Remove H_Pareto from contributions** until formula is fixed
4. **Don't overclaim efficient coding as pre-registered** - H2 is a failed hypothesis
5. **Add H_Comp results to paper** or explain why omitted