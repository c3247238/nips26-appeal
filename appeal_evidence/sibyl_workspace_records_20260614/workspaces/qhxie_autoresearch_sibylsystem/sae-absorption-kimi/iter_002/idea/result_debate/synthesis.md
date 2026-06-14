# Result Debate Synthesis: SAEBench Absorption Construct-Validity Study

## 1. Consensus Map

All six perspectives converge on the following high-confidence conclusions:

1. **The primary correlation (H1) is underpowered and inconclusive.** With only n=7 SAEs, the Pearson r = 0.463 and its 95% CI [-0.389, 0.981] cannot support or reject the construct-validity claim. This is unanimously agreed.

2. **Architecture ranking is partially preserved.** The ordinal ordering PAnneal < Matryoshka < Gated < JumpRelu < TopK < BatchTopK is consistent across first-letter and semantic-hierarchy tasks. This suggests the metric retains *some* ordinal utility for comparing architectures, even if absolute construct validity is weak.

3. **The tau_fs robustness finding is technically sound but evidentially vacuous.** The correlation is remarkably stable across thresholds (0.463–0.471), but because the estimate itself is imprecise, stability does not imply validity.

4. **Two results are genuinely surprising and publication-quality:** (a) semantic-hierarchy absorption is *lower* than non-hierarchy control absorption (reversed H2), and (b) a Random SAE matches the Standard SAE on semantic tasks. These are not disputed by any perspective.

5. **The GPT-2 replication is anomalous and uninterpretable without further validation.** Near-zero absorption (0.0, 0.003) on GPT-2 small, versus ~0.35 on Pythia-160M, is so discrepant that it demands diagnosis before it can support any model-generalizability claim.

---

## 2. Conflict Resolution

### Conflict 1: Is the semantic-hierarchy measurement "broken" (skeptic/methodologist) or merely "weak" (optimist/strategist)?

**Resolution:** The skeptic and methodologist raise fatal-severity concerns about perfect probe AUROCs (1.0) and the failed Random-SAE control. These are not hyperbolic—they directly undermine the interpretation of the semantic-hierarchy score as a measure of "absorption." However, the optimist and strategist correctly note that even if the metric is confounded, the *patterns* it produces (ranking preservation, reversed specificity, model-dependence) are still publishable as a diagnostic study. The measurement is not so broken that the patterns are random noise; it is broken in systematic ways that are themselves interesting.

**Judgment call:** The semantic-hierarchy task is likely too easy (ceiling-effect probes) and the absorption formula is not specific to learned structure on this task (Random ≈ Standard). The negative results are therefore *partially interpretable* as boundary-condition findings, but the authors must not claim them as a clean falsification of construct validity without first validating the probe.

### Conflict 2: Should the paper be reframed as a "construct-validity stress-test" (strategist/comparativist) or does it need a full measurement fix before publication (skeptic/methodologist)?

**Resolution:** The strategist and comparativist argue that the reversed H2 and random-SAE results are already compelling enough for a workshop or short paper. The skeptic and methodologist counter that a paper claiming "lack of construct validity" must first demonstrate that the measurement instrument is valid. These positions are reconcilable: the paper should be framed as a **diagnostic benchmark study** that identifies boundary conditions and measurement artifacts, *not* as a definitive falsification. This framing satisfies the strategist's narrative while respecting the skeptic's epistemic guardrails.

### Conflict 3: Does the GPT-2 result represent a genuine model-family effect (optimist/revisionist) or a pipeline bug (methodologist)?

**Resolution:** The methodologist's concern is stronger. A drop from 0.35 to 0.0 is larger than any architecture effect in the entire study and occurs with only 2 SAEs at a different layer. Until the pipeline is audited (hook names, SAE releases, dictionary sizes), the GPT-2 result should be treated as **unvalidated** and reported as a pilot anomaly rather than a confirmed finding.

---

## 3. Result Quality Score: 5 / 10

**Justification:** The study has a clear, high-stakes research question and a rigorous experimental design (multiple architectures, controlled conditions, bootstrap CIs). However, the primary hypothesis is inconclusive due to small sample size, the semantic-hierarchy probe shows signs of degeneracy (perfect AUROCs), and the Random-SAE control fails a pre-registered expectation. The most interesting findings are the *unexpected* ones (reversed H2, Random ≈ Standard), but these challenge the original hypothesis rather than confirming it. The GPT-2 replication is currently uninterpretable. The score is saved by the novelty of the question and the honesty of the negative results, but it is held back by measurement validity concerns that must be addressed before a top-tier submission.

---

## 4. Key Findings

1. **First-letter absorption does not robustly predict semantic-hierarchy absorption.** The correlation is moderate (r = 0.463) but the confidence interval is too wide to support the construct-validity claim.

2. **The absorption metric is not hierarchy-specific.** Non-hierarchy correlated pairs (doctor-hospital, happy-joyful) produce *higher* absorption scores than true WordNet hypernym hierarchies, suggesting the metric is more sensitive to co-occurrence/correlation structure than to hierarchical containment.

3. **A random permuted-decoder SAE matches the Standard SAE on semantic tasks.** This indicates the semantic-hierarchy absorption score is not specific to learned SAE structure and may reflect base-model representational geometry or probe-projection artifacts.

4. **Architecture ranking is preserved across tasks.** Despite weak absolute correlation, the relative ordering of SAEs by absorption is consistent across first-letter and semantic tasks, suggesting the metric retains ordinal utility for architecture comparison.

5. **The GPT-2 replication shows near-zero absorption, but this result is unvalidated.** The dramatic divergence from Pythia-160M demands pipeline diagnosis before it can support any model-generalizability conclusion.

---

## 5. Methodology Gaps

From the methodologist and skeptic, the following gaps are critical:

1. **Probe degeneracy (AUROC = 1.0).** Every hierarchy probe achieves perfect discrimination on the base model. This suggests the task is trivially easy and may distort the absorption formula via ceiling effects. *Remediation:* re-test with harder hierarchies that yield base-model AUROCs in the 0.75–0.95 range.

2. **Failed Random-SAE control.** The permuted-decoder baseline scores ~0.35 on semantic hierarchies, identical to Standard SAE. This falsifies the pre-registered assumption that the metric is specific to learned structure. *Remediation:* test 3 independently randomized SAEs; if all score ~0.35, the semantic task must be redesigned or the formula re-examined.

3. **Small-sample correlation (n=7).** The primary H1 is underpowered. *Remediation:* increase effective n by evaluating multiple checkpoints per family, multiple layers, or adding architectures (OrtSAE, HSAE).

4. **No probe-AUROC standardization for hierarchy specificity.** The paired t-test uses raw absorption scores without adjusting for probe difficulty. *Remediation:* compute `standardized_absorption = raw_absorption / probe_AUROC` and re-test.

5. **Unvalidated GPT-2 replication.** The near-zero GPT-2 scores are so anomalous that a pipeline bug cannot be ruled out. *Remediation:* audit hook names, SAE releases, and layer indices; re-run if necessary.

6. **Missing ablations.** Natural-frequency (unmatched) baseline, multi-token vs single-token comparison, and k-sparse probe ablation were proposed but not executed.

---

## 6. Competitive Position vs. SOTA

The comparativist's assessment is adopted in full:

- **This is not a "beat SOTA" paper.** It is a **methodological diagnostic** paper. The contribution margin is measured by the novelty of the question and the implications of the answer, not by percentage improvement.
- **Novelty is genuine but narrow:** this is the first study to empirically test whether the dominant SAEBench absorption metric generalizes from first-letter tasks to semantic hierarchies.
- **No direct concurrent overlap** exists, though multiple 2025–2026 papers are questioning SAEBench metric validity from other angles (synthetic data, OOD generalization). The window for this narrative may close within 6–12 months.
- **Venue recommendation:** ICLR / NeurIPS MI Workshop as-is; EMNLP Findings or AAAI with a Gemma-2-2B replication and additional ablations. Top-tier main conference (NeurIPS/ICML/ICLR) is unlikely without stronger empirical validation.

---

## 7. Hypothesis Update

From the revisionist:

| Hypothesis | Updated Verdict |
|---|---|
| H1: Construct validity (first-letter → semantic) | **Inconclusive / weakly negative.** The correlation is too imprecise to support the claim. |
| H2: Hierarchy specificity | **Refuted.** The metric is more sensitive to co-occurrence than to hierarchy. |
| H3: Robustness across tau_fs | **Inconclusive.** Stable but imprecise. |
| H4: Model generalization | **Inconclusive.** GPT-2 anomaly demands validation. |
| H5: Architecture ordering | **Partially supported.** Ordinal ranking is preserved, but Standard SAE and Random SAE collapse together on semantic tasks. |

**Core mental-model update:** First-letter absorption is not a general measure of "feature absorption." It is a task-specific metric that captures how well an SAE avoids splitting *character-level* features, which may be only weakly related to semantic-level hierarchies. The high scores on random decoders and the inversion of hierarchy vs. non-hierarchy specificity indicate that the metric is confounded by base-model geometry and general correlation structure.

---

## 8. Action Plan

### Verdict: **PROCEED with reframed narrative**

The project should **not pivot.** The unexpected findings (reversed H2, Random ≈ Standard) are publication-quality and more interesting than a confirmatory result would have been. However, the narrative must shift from "construct validity confirmed" to **"A Construct-Validity Stress-Test of the SAEBench Absorption Metric."**

### Prioritized Next Steps

| Priority | Experiment | GPU Hours | Rationale |
|---|---|---|---|
| 1 | **Diagnose GPT-2 replication anomaly** | ~0.1 (audit + possible re-run) | The near-zero GPT-2 scores are so discrepant that they cast doubt on the custom pipeline. Must validate before any model-generalizability claim. |
| 2 | **Weak co-occurrence non-hierarchy control** | ~0.3 | Tests the leading explanation for reversed H2. If weak pairs show lower absorption, the co-occurrence hypothesis is confirmed. |
| 3 | **Re-test with non-trivial hierarchies (AUROC < 1.0)** | ~0.3 | Addresses the fatal probe-degeneracy concern. If the same patterns hold on harder hierarchies, the findings are strengthened. |
| 4 | **Add OrtSAE + HSAE to architecture ranking** | ~0.3 | Increases n from 7 to 9, tightening H1 CI. Also tests whether ranking preservation generalizes to newer absorption-reducing designs. |
| 5 | **Standardize absorption by probe AUROC** | ~0.1 (re-analysis) | Confounds probe-difficulty from hierarchy-specificity test. Low cost, high credibility impact. |
| **Total** | | **~1.1** | Well within the 1-hour-per-iteration guideline. |

### Narrative Reframe (Final)

**Old story:** "First-letter absorption predicts semantic-hierarchy absorption (construct validity confirmed)."

**New story:** **"A Construct-Validity Stress-Test of the SAEBench Absorption Metric."**

Key claims:
1. **Ordinal validity holds:** Architecture rankings are preserved across first-letter and semantic tasks.
2. **Absolute validity is weak:** Correlation is moderate and underpowered (r = 0.46, wide CI).
3. **Hierarchy specificity fails:** The metric detects co-occurrence better than true hierarchy.
4. **Random baselines are competitive:** A permuted-decoder SAE matches trained SAEs on semantic tasks.
5. **Model dependence is large (pending validation):** GPT-2 shows near-zero absorption where Pythia shows substantial absorption.

This framing delivers both a methodological contribution and a critical insight—exactly the kind of paper that advances the field by identifying boundary conditions rather than confirming assumptions.
