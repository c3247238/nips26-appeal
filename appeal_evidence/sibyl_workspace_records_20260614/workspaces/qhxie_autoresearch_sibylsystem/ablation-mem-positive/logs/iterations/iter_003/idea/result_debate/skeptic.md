# Skeptic Analysis: CV Predicts Steering Heterogeneity in Absorbed SAE Features

## 1. Statistical Risk Inventory

### Risk 1: CV Threshold (1.0) Is Post-Hoc — Potential Overfitting

**Citation**: `proposal.md` Section 4 — "CV > 1.0 threshold was chosen based on pilot data distribution"

The CV > 1.0 threshold was selected after observing the pilot results showing 0.153 vs 0.075 (2x difference). This is classic post-hoc threshold optimization. A prospective threshold should have been specified before collecting the full experiment data, or at minimum, a principled information-theoretic derivation should justify 1.0 rather than 0.5, 2.0, or any other value.

**Consequence**: If the threshold were tuned to maximize the high-CV vs low-CV split, the reported p-values are inflated. The "significant at p < 0.01" claim is only valid if the threshold was specified before data collection.

**What would resolve this**: A held-out feature set where the CV threshold was specified before seeing steering results. Alternatively, a theoretical derivation of the optimal threshold from first principles.

---

### Risk 2: Low-CV Features Are Extremely Rare — Sampling Bias

**Citation**: `optimist.md` Finding 1 — "Among 8354 absorbed features, 98.7% (8248) are high-CV while only 1.3% (106) are low-CV"

With only 106 low-CV features in the entire population, our experiment tested 30 low-CV features (28% of the total population). The low-CV group is not merely small — it is a rare, potentially anomalous subpopulation. These 106 features might all be "edge cases" that were absorbed for unusual reasons, making them a systematically biased sample.

**Consequence**: The steering effect we measure for low-CV features (0.21-0.50 across strengths) may not generalize to any low-CV feature because the 106 features are not representative of a coherent "low-CV absorbed" category.

**What would resolve this**: Test all 106 low-CV features to confirm the current result, or explicitly acknowledge this as a limitation and present it as "preliminary" rather than "validated."

---

### Risk 3: p-values Reporting as 0.0 Suggests Numerical Underflow

**Citation**: `full_steering_cv.json` — "p_value_one_sided": 0.0 for all three strengths; `optimist.md` — "p=0.0 (all strengths)"

The t-distribution with n=298 degrees of freedom does not actually produce p=0.0 — this is numerical underflow. The real p-value is something like 10^-15 or smaller, but reporting it as 0.0 is misleading. More importantly, with such extremely small p-values, the practical significance (effect size) matters more than the statistical significance.

**Consequence**: The BH-corrected significant results are real, but the "0.0" reporting obscures the actual magnitude. We should be reporting actual p-values (e.g., "p < 10^-15") or at minimum, acknowledging the underflow.

---

## 2. Alternative Explanations

### Alternative 1: CV Correlates with Activation Magnitude, Not Genuine Variability

The coefficient of variation (CV = std/mean) normalizes by the mean. But if high-CV features systematically have higher activation variance AND higher mean activation, the ratio might be measuring something about overall feature magnitude rather than "coefficient of variation" in the theoretically meaningful sense. Fano factor (CV²/mean) was proposed as a control but the proposal acknowledges this was "PENDING" — we don't know if the CV-steering correlation survives magnitude control.

**Test**: Compute Fano factor-normalized CV and re-test the correlation. If the correlation disappears after controlling for activation magnitude, CV is merely a proxy for feature "strength" rather than a genuine predictor.

---

### Alternative 2: The "Parity with Non-Absorbed" Finding Is Selection Bias

**Citation**: `optimist.md` Finding 2 — "Non-absorbed: 0.1020 vs Absorbed high-CV: 0.0975 (only 4.5% difference)"

This parity is presented as evidence that absorbed high-CV features "retain steering potential." But consider: absorbed features that are high-CV might be the ones that "survived" absorption with their causal structure intact — the features that lost steering potential may have been absorbed in a way that destroys it, and they might be low-CV. This would create selection bias where the absorbed high-CV subpopulation is enriched for steerable features by definition.

**Test**: Compare non-absorbed features matched on activation magnitude and frequency. If absorbed high-CV features show parity after matching, the selection bias explanation is weakened.

---

### Alternative 3: The CV-Steering Correlation Is Architecture-Specific

All results are on GPT-2 small with TopK SAEs. The CV distribution among absorbed features (98.7% high-CV) is so extreme that it might be specific to how TopK SAEs absorb features on GPT-2. Gemma-2's JumpReLU SAEs may show completely different absorption dynamics — different CV distributions or different CV-steering relationships.

**Test**: Cross-architecture replication is listed as "PENDING" — this is the single most important validation needed before claiming generality.

---

## 3. Proxy Metric Audit

### Gap: Logit Change ≠ Interpretability Utility

The experiment measures logit change at target tokens, but the paper's claims are about "steering effectiveness" and "practical interpretability utility." The connection between these is not established:

- A logit change of 0.3 might be meaningful for some features but negligible for others depending on the semantic sensitivity of the output space
- We don't know if the ~47% improvement in logit change translates to proportionally better downstream task performance or human interpretability
- The "actionability paradox" from Basu et al. was about clinical utility — we haven't tested whether high-CV steering actually helps with clinically meaningful tasks

**What would resolve this**: Test steering on a downstream task (e.g., sentiment manipulation, fact retrieval) to validate that logit changes translate to functional utility.

---

## 4. Severity Classification

### Fatal Flaw: None identified

The core finding (CV predicts steering heterogeneity) is robustly confirmed with consistent effect sizes across three steering strengths. The statistical evidence is strong even if the p-values are reported in misleading form.

### Serious Concern 1: Post-hoc CV threshold inflates significance

**Severity**: Serious concern — the threshold should have been prospectively specified
**Remediation**: Either (a) acknowledge the threshold was post-hoc and present results as exploratory, or (b) validate on a held-out feature set with pre-specified threshold

### Serious Concern 2: Low-CV sampling bias undermines generalizability

**Severity**: Serious concern — the 30 low-CV features tested may not represent the full low-CV population
**Remediation**: Test all 106 low-CV features; if that's not feasible, acknowledge the limitation transparently

### Serious Concern 3: Cross-architecture validation is pending

**Severity**: Serious concern — all results are on GPT-2/TopK SAEs; generalizability is unconfirmed
**Remediation**: Complete Gemma-2 replication before claiming the finding generalizes beyond GPT-2

### Minor Caveat: Mechanism remains unknown

Decoder orthogonality was ruled out as an explanation, but no alternative mechanism has been identified. This is a genuine knowledge gap but does not invalidate the empirical finding.

---

## 5. Concrete Remediation Experiments

| Issue | Experiment | Dataset | Metric | Expected Outcome |
|-------|-----------|---------|--------|-----------------|
| Post-hoc threshold | Held-out validation: split features 50/50, specify threshold before running on hold-out set | GPT-2 layer 6 | High-CV vs low-CV steering effect | Similar effect size on hold-out confirms threshold is not overfitted |
| Low-CV sampling bias | Test all 106 low-CV features (not just 30) | GPT-2 layer 6 | Mean steering effect for all 106 vs 30 sampled | If 106 features show different effect, current estimate is biased |
| Mechanism | Fano factor normalization: compute CV²/mean and retest correlation | GPT-2 layer 6 | Partial correlation of Fano factor with steering | If correlation drops to ~0, CV is proxy for magnitude |
| Cross-architecture | Replicate full_steering_cv on Gemma-2-2B JumpReLU SAE | Gemma-2-2B layer 6 | High-CV vs low-CV steering effect | Similar 40-50% improvement confirms generalizability |
| Practical utility | Downstream task steering: test CV-based feature selection on sentiment manipulation | SST-2 or similar | Task accuracy improvement from steering | If steering improves task performance proportionally, validates practical utility |

---

## 6. Bottom Line

**The core finding is real but the scope of confidence is narrower than claimed.**

What IS confirmed:
- High-CV absorbed features show ~47% larger steering effects than low-CV absorbed features on GPT-2/TopK SAEs
- The effect is consistent across three steering strengths and statistically significant

What is NOT yet confirmed:
- The CV > 1.0 threshold is not validated as optimal — it could be overfitted to pilot data
- The low-CV steering estimate is based on a small, potentially biased sample (28% of population)
- Generalizability beyond GPT-2/TopK SAEs is unconfirmed — Gemma-2 replication is essential

**The paper should be positioned as "preliminary evidence for CV-based decomposition of absorbed features" rather than "validated finding with generalizable conclusions."** The honest framing is: we found a promising predictor on GPT-2 that warrants cross-architecture validation before claiming it resolves or refines the actionability paradox.

---