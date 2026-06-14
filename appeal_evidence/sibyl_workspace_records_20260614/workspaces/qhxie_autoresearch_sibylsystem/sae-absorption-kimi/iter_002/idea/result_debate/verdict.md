# Verdict: SAEBench Absorption Construct-Validity Study

## Result Quality Score: 5 / 10

The study asks a high-stakes, novel question but returns an inconclusive primary result alongside two unexpected findings that challenge the original hypothesis. The experimental design is rigorous, but measurement validity concerns (perfect probe AUROCs, failed Random-SAE control) and an unvalidated GPT-2 anomaly prevent a higher score.

---

## Key Conclusion

**First-letter absorption is not a general measure of "feature absorption."** The data support a diagnostic reframing: the SAEBench absorption metric captures task-specific representational properties (likely character-level feature splitting and base-model geometry) that do not generalize cleanly to semantic hierarchies. The most compelling findings are the *failures* of the original hypothesis: hierarchy specificity is reversed (non-hierarchy > hierarchy), and a random permuted-decoder SAE matches trained SAEs on semantic tasks. These are not fatal flaws of the study—they are the study's contribution, provided the narrative is reframed accordingly.

---

## What We Actually Learned

1. **The primary correlation (H1) is underpowered and inconclusive.** With n=7 SAEs, r = 0.463 and CI [-0.389, 0.981] cannot support or reject construct validity.
2. **The metric is not hierarchy-specific.** True WordNet hypernyms produce *lower* absorption scores than non-hierarchy correlated pairs, suggesting the formula is more sensitive to co-occurrence than to hierarchical structure.
3. **Random decoders score as high as Standard SAEs on semantic tasks.** This indicates the semantic-hierarchy score is not specific to learned SAE structure.
4. **Architecture ranking is partially preserved.** The ordinal ordering across SAE families is consistent across tasks, suggesting the metric retains *some* utility for comparative architecture evaluation.
5. **The GPT-2 replication is anomalous and demands validation.** Near-zero absorption on GPT-2 (vs. ~0.35 on Pythia) is so discrepant that a pipeline bug cannot be ruled out.

---

## PIVOT or PROCEED?

**RECOMMENDATION: PROCEED with narrative reframe.**

Do **not** pivot. The unexpected findings are publication-quality and more interesting than a confirmatory result would have been. The paper should be reframed from "construct validity confirmed" to **"A Construct-Validity Stress-Test of the SAEBench Absorption Metric."**

This framing:
- Acknowledges the ordinal utility of the metric (ranking preservation)
- Reports the weak absolute correlation honestly
- Highlights the boundary conditions (co-occurrence sensitivity, random-SAE floor, model dependence)
- Positions the work as a methodological diagnostic rather than a failed confirmation

---

## Action Plan

| Priority | Action | GPU Hours | Expected Outcome |
|---|---|---|---|
| 1 | **Diagnose GPT-2 replication anomaly** | ~0.1 | Confirm whether the near-zero GPT-2 scores are genuine or a pipeline bug. |
| 2 | **Run weak co-occurrence non-hierarchy control** | ~0.3 | Test the co-occurrence hypothesis for reversed H2. |
| 3 | **Re-test with non-trivial hierarchies (AUROC < 1.0)** | ~0.3 | Address probe-degeneracy concern and strengthen credibility. |
| 4 | **Add OrtSAE + HSAE to architecture comparison** | ~0.3 | Increase n, tighten H1 CI, test ranking generalization. |
| 5 | **Standardize absorption scores by probe AUROC** | ~0.1 (re-analysis) | Remove probe-difficulty confound from hierarchy-specificity test. |
| **Total** | | **~1.1** | |

**Target venue:** ICLR / NeurIPS MI Workshop as-is; EMNLP Findings / AAAI with Gemma-2-2B replication and additional ablations. Top-tier main conference unlikely without stronger validation.
