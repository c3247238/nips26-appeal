# Result Debate Verdict

**Date**: 2026-04-14 | **Iteration**: 004 | **Synthesizer**: sibyl-result-synthesizer

---

## Result Quality Score: 5.5 / 10

The study delivered one genuinely strong finding amid several honest falsifications. The ambitious theoretical framework (Lotka-Volterra competitive exclusion) is empirically dead, but the downstream correlation analysis -- which the proposal identified as the "best possible outcome" -- exceeded all expectations.

---

## Key Conclusion

**Absorption matters.** Across 54 Gemma-2-2B SAEs, absorption score is a strong negative predictor of downstream SAE quality: r = -0.595 (sparse probing F1), -0.431 (SCR), -0.454 (RAVEL TPP). After controlling for width, layer, and architecture, partial correlations *strengthen* to r = -0.661 to -0.677. A matched-pair comparison confirms the relationship with Cohen's d = 2.13 (p = 0.006). This is the first systematic, statistically controlled evidence that SAE absorption predicts downstream task degradation -- directly addressing the DeepMind concern that motivated this research.

The LV competition coefficient (H1) and corpus PMI predictor (H2) are both cleanly falsified. The absorption taxonomy reveals that the true scope of absorption likely exceeds reported rates, but the specific 92.3% figure requires methodological recalibration before publication. The 30-SAE scaling survey (layer gradient, width scaling) provides novel empirical contributions.

---

## Action Plan

### Decision: **PROCEED** with major framing pivot

Drop the Lotka-Volterra theoretical framework as the paper's centerpiece. Restructure around the downstream impact finding as the primary contribution.

### Priority 1 (Immediate, 0 GPU-hours)
- Add L0 as covariate in C3A partial correlations (address the biggest confound threat)
- Compute within-width stratified correlations (eliminate width confound objection)

### Priority 2 (Near-term, 3 GPU-hours)
- Recalibrate taxonomy with sae-spelling ground-truth parent features
- Scale safety probe to n >= 10 SAEs at fixed layer

### Priority 3 (Parallel, 8 GPU-hours)
- Replicate key findings on Gemma-2-2B (essential for NeurIPS-tier venue)

### Priority 4 (Immediate, 0 GPU-hours)
- Begin paper draft with pivoted framing: "Empirical Anatomy of Feature Absorption: Prevalence, Scaling, and Downstream Impact"

---

## Revised Paper Contributions (Ranked)

1. First systematic evidence that absorption predicts downstream SAE quality (r = -0.60 to -0.68)
2. Comprehensive absorption taxonomy (Type I/II/III) showing true rates exceed Chanin metric
3. 30-SAE scaling survey: layer gradient and width scaling laws for absorption
4. Informative negative results: LV detector and corpus PMI predictor falsified

## Venue Target

- **With L0 control + taxonomy fix + Gemma-2 replication**: NeurIPS 2026 main conference
- **With L0 control + taxonomy fix only**: NeurIPS 2026 workshop / AAAI main conference
- **Current state (no additional work)**: Workshop paper only
