# Skeptic Analysis: Construct Validity of SAEBench Absorption

## Executive Summary

The experiment produces a **failed construct-validity test** and a **statistically significant but directionally wrong** hierarchy-specificity test. The main claim—that first-letter absorption predicts semantic-hierarchy absorption—is **not supported** by the data. Several measurement artifacts and design choices cast doubt on whether the negative result reflects a true lack of construct validity or simply a poorly calibrated semantic-hierarchy probe. Below I inventory the statistical risks, alternative explanations, and proxy-metric gaps.

---

## 1. Statistical Risk Inventory

### Risk 1: Tiny sample size makes the correlation estimate almost uninformative
- **Specific number:** n = 7 SAEs (excluding Random) for the primary H1 correlation.
- **Why it is unreliable:** A Pearson r = 0.463 with n = 7 yields a bootstrap 95% CI of [-0.389, 0.981]. The interval spans from moderate negative to near-perfect positive. With this width, the study has virtually no power to distinguish r = 0.6 (the pre-registered success threshold) from r = 0.0. The pre-registered falsification rule (CI includes values < 0.3) is met, but so is the support rule in the opposite tail—the result is pure noise.
- **Severity:** **Serious concern**. The correlation point estimate is reported as if it carries signal, but the confidence interval shows it is consistent with almost any relationship.

### Risk 2: Perfect probe AUROCs (1.0 for all 10 hierarchies across all 8 SAEs) signal a degenerate probe, not a strong one
- **Specific number:** Every entry in `per_hierarchy_probe_auroc` is exactly 1.0.
- **Why it is unreliable:** A perfect AUROC on every concept and every SAE seed is implausible for logistic probes trained on residual-stream activations of a 160M-parameter model. This strongly suggests the probe has overfit or the dataset construction has leaked labels (e.g., parent/child tokens are perfectly separable by a trivial feature like token ID or position). If the probe is degenerate, the absorption formula downstream receives a manipulated input and the resulting scores are uninterpretable.
- **Severity:** **Fatal flaw** (conditional on validation). If the probe is indeed degenerate, the entire semantic-hierarchy absorption measurement is invalidated.

### Risk 3: The Random-SAE control fails the pre-registered expectation
- **Specific number:** Random SAE semantic-hierarchy absorption = 0.352; non-hierarchy control = 0.416.
- **Why it is unreliable:** The proposal states: "Should yield near-zero absorption on all tasks; if not, the metric is not specific to learned structure." Random SAE scores are not near-zero; they are numerically indistinguishable from Standard SAE (0.352 vs. 0.352 semantic; 0.416 vs. 0.416 control). This means the metric is **not specific to learned structure** on the semantic tasks. The experiment therefore measures something other than absorption—possibly probe difficulty, dataset bias, or a mathematical tautology in the formula when applied to these concepts.
- **Severity:** **Fatal flaw**. The control condition explicitly falsifies the assumption that the semantic-hierarchy task measures the same phenomenon as first-letter absorption.

---

## 2. Alternative Explanations

### For the low first-letter vs. semantic-hierarchy correlation (r = 0.463, CI includes 0)

**Alternative A: The semantic-hierarchy probe is not measuring the same representational geometry as the first-letter probe.**
- First-letter tasks use ground-truth probes on a synthetic classification task with clear linear structure in token space. Semantic hierarchies in Pythia-160M may not be linearly separable in the residual stream at layer 8, or may be represented via distributed/contextual features rather than a single linear direction. The probe may be fitting to spurious correlates (e.g., co-occurrence statistics) rather than the true hypernymy relationship. If so, the SAEBench absorption formula is applied to a mismatched "ground-truth" direction, and the resulting score is uncorrelated with first-letter absorption by construction.

**Alternative B: The SAE cohort is too homogeneous on the semantic task to detect variance.**
- All semantic-hierarchy absorption scores cluster in a narrow band (0.064–0.359, excluding Random). The coefficient of variation is low. If the task is simply "easy" for all SAEs (or the probe is degenerate), there is no variance to correlate with first-letter absorption. The null correlation could reflect a ceiling/floor effect rather than a lack of construct validity.

### For the hierarchy-specificity result (semantic mean = 0.235 < non-hierarchy mean = 0.331, p = 0.003)

**Alternative A: The non-hierarchy concepts were harder to probe, inflating their absorption scores through probe-difficulty confounding.**
- The proposal explicitly identifies this threat and proposes AUROC-standardization as a mitigation, but the reported results are raw absorption scores with no standardization. If non-hierarchy probes had lower AUROCs (we cannot verify this because probe AUROCs are not reported for the control condition), higher absorption could be an artifact of weaker ground-truth directions. The significant p-value would then reflect a probe-quality difference, not a true lack of hierarchy specificity.

**Alternative B: The non-hierarchy pairs were not properly frequency-matched.**
- Frequency matching is claimed for all conditions, but no per-pair frequency table is provided in the summary. If non-hierarchy tokens had more imbalanced frequencies than hierarchy tokens, the SAEBench absorption formula (which is sensitive to frequency imbalance) could produce higher scores for the control condition. The significant difference would then be a dataset-construction artifact.

---

## 3. Proxy Metric Audit

| What we claim | What we actually measure | Gap |
|---|---|---|
| "Construct validity of first-letter absorption" | Pearson correlation between first-letter and semantic-hierarchy absorption across 7 SAEs | The semantic-hierarchy task may not measure absorption at all (Random SAE scores ≈ Standard SAE scores). |
| "Hierarchy specificity" | Paired t-test of raw absorption scores (semantic vs. non-hierarchy) | No probe-AUROC standardization applied; non-hierarchy probe quality unknown. |
| "Robustness across tau_fs" | Correlation at three thresholds with identical n=7 sample | CIs remain astronomically wide; robustness claim is vacuous. |
| "Generalization to semantic hierarchies" | 10 WordNet hypernym pairs on Pythia-160M layer 8 | Probe AUROCs are all exactly 1.0, suggesting the task is trivial or degenerate; generalization claim is unsupported. |

**Key finding:** The gap between "what we measure" and "what we claim" is largest for the semantic-hierarchy absorption score itself. Because the Random-SAE control fails, we do not have evidence that the semantic-hierarchy score measures absorption at all. It may simply measure how often a logistic probe with AUROC = 1.0 projects onto SAE latents in a way that satisfies the absorption formula—a mathematical consequence unrelated to learned hierarchical structure.

---

## 4. Severity Classification

| Issue | Severity | Rationale |
|---|---|---|
| Perfect probe AUROCs (1.0) on all hierarchies | **Fatal flaw** | Invalidates the assumption that the probe is measuring a non-trivial representational direction. |
| Random SAE scores ≈ Standard SAE scores on semantic tasks | **Fatal flaw** | Directly falsifies the pre-registered control expectation; the metric is not specific to learned structure. |
| n = 7 for primary correlation with CI spanning [-0.39, 0.98] | **Serious concern** | The correlation estimate is too imprecise to support any claim about construct validity. |
| No AUROC-standardization or probe-quality table for control condition | **Serious concern** | The hierarchy-specificity test is confounded by probe difficulty; the significant p-value is uninterpretable. |
| GPT-2 replication shows near-zero absorption on both Standard and TopK | **Minor caveat** | This is actually a well-reported negative result, but it raises the question of whether the semantic task is too easy/hard on different models, adding model-specificity concerns. |
| Pilot showed expected Matryoshka < TopK ordering, but full experiment does not replicate cleanly | **Serious concern** | Pilot results (Matryoshka 0.283, TopK 0.339) do not generalize to the full cohort ranking; this suggests pilot optimism or cherry-picking. |

---

## 5. Concrete Remediation

### For the fatal flaws

**Remediation 1: Audit and fix the semantic-hierarchy probe.**
- **Experiment:** Re-train probes with regularization (L2 penalty, smaller max_iter) and explicit train/val splits. Report per-hierarchy validation AUROCs. If any hierarchy still shows AUROC = 1.0, inspect the dataset for label leakage (e.g., parent/child tokens are the only tokens in their respective classes).
- **Expected outcome:** Validation AUROCs in the 0.7–0.95 range. If they remain at 1.0, redesign the dataset (e.g., add distractor tokens, use contextual rather than single-token classification).

**Remediation 2: Validate the Random-SAE control.**
- **Experiment:** Compute semantic-hierarchy absorption on 3 independently randomized SAEs (different decoder permutations). If all three yield scores near-zero (< 0.05), the metric is validated. If they continue to yield scores ~0.35, the absorption formula itself is not suitable for this task.
- **Expected outcome:** Either the control passes (metric is salvageable) or it fails (the semantic task must be redesigned or abandoned).

### For the serious concerns

**Remediation 3: Increase effective sample size or tighten the estimation target.**
- **Experiment:** Instead of a single correlation across 7 SAE families, compute the correlation across **individual SAE checkpoints** (e.g., multiple trainers per family, or multiple layers). This could increase n from 7 to 15–30. Alternatively, pre-register a **non-inferiority bound** (e.g., test whether r > 0.3 rather than r > 0.6) to match the available power.
- **Expected outcome:** A correlation estimate with a CI width < 0.4, enabling a conclusive claim.

**Remediation 4: Standardize absorption scores by probe AUROC before testing hierarchy specificity.**
- **Experiment:** For each SAE and each concept pair, compute `standardized_absorption = raw_absorption / probe_AUROC`. Re-run the paired t-test on standardized scores. Also report per-condition probe AUROC tables.
- **Expected outcome:** If the significant difference disappears after standardization, the original result was a probe-difficulty artifact. If it persists, the hierarchy-specificity claim is strengthened.

**Remediation 5: Report per-pair token frequencies for both conditions.**
- **Experiment:** Append a table showing the empirical frequency ratio of parent:child tokens in the synthetic dataset for every pair in both hierarchy and non-hierarchy conditions. Confirm that ratios are balanced (≈ 1:1) and that non-hierarchy pairs are not more imbalanced.
- **Expected outcome:** Frequency ratios within 0.8–1.2 for all pairs. If not, re-sample the datasets.

---

## 6. Bottom Line

The experiment does **not** support the claim that first-letter absorption generalizes to semantic hierarchies. However, the skeptic's verdict is that this negative result is **currently uninterpretable** because two fatal flaws—perfect probe AUROCs and a failed Random-SAE control—suggest the semantic-hierarchy measurement is broken, not that the construct-validity hypothesis is false. Before any paper can claim a "lack of construct validity," the authors must demonstrate that their semantic-hierarchy task actually measures absorption in the first place.
