# Testable Hypotheses

## Primary Hypothesis (H1): Construct Validity

**Statement:** SAEs with low first-letter absorption rates (as measured by SAEBench) will also exhibit low absorption rates on matched-frequency semantic hierarchies drawn from WordNet. The Pearson correlation between first-letter absorption scores and semantic-hierarchy absorption scores across a diverse set of 6–8 SAEs will be greater than 0.6.

**Falsification criterion:** If the Pearson correlation is below 0.6 (or non-significant at p > 0.05), the hypothesis is falsified. This would imply that optimizing first-letter absorption may not improve SAE behavior on real-world hierarchical features.

**Expected outcome if true:** r > 0.6 with a 95% bootstrap CI that excludes 0.
**Expected outcome if false:** r < 0.6 or CI includes 0.

---

## Secondary Hypothesis (H2): Hierarchy Specificity

**Statement:** The SAEBench absorption metric is specific to hierarchical features rather than merely detecting correlated-feature co-occurrence. Semantic-hierarchy absorption scores will be significantly higher than non-hierarchy correlated-feature absorption scores.

**Falsification criterion:** If the paired t-test between semantic-hierarchy and non-hierarchy control absorption scores is non-significant (p > 0.05), the hypothesis is falsified. This would suggest the metric detects general correlation rather than hierarchy.

**Expected outcome if true:** Semantic-hierarchy absorption > non-hierarchy control absorption, p < 0.05.
**Expected outcome if false:** No significant difference between the two conditions.

---

## Tertiary Hypothesis (H3): Robustness Across Thresholds

**Statement:** The correlation between first-letter and semantic-hierarchy absorption is robust to the choice of feature-splitting threshold τ_fs.

**Falsification criterion:** If the correlation sign flips or drops below 0.3 when τ_fs is varied from 0.01 to 0.05, the hypothesis is falsified.

**Expected outcome if true:** r remains positive and > 0.5 across τ_fs ∈ {0.01, 0.03, 0.05}.
**Expected outcome if false:** r becomes negative or near-zero at one or more threshold values.

---

## Exploratory Hypothesis (H4): Model Generalization

**Statement:** The correlation pattern between first-letter and semantic-hierarchy absorption replicates across different base models (Gemma-2-2B and GPT-2 small).

**Falsification criterion:** If the correlation signs differ between models, or if one model shows r > 0.6 while the other shows r < 0.3, the hypothesis is falsified.

**Expected outcome if true:** Both models show positive correlations in the same direction, with overlapping bootstrap CIs.
**Expected outcome if false:** Divergent or opposite correlation patterns across models.

---

## Exploratory Hypothesis (H5): Architecture Ordering

**Statement:** The relative ranking of SAEs by absorption rate is preserved across first-letter and semantic-hierarchy tasks. Low-absorption architectures (Matryoshka, OrtSAE) remain low-absorption on semantic hierarchies, and high-absorption architectures (Standard ReLU, JumpReLU) remain high-absorption.

**Falsification criterion:** If the architecture ordering by absorption rate inverts between first-letter and semantic-hierarchy tasks (e.g., Matryoshka shows higher semantic-hierarchy absorption than Standard ReLU), the hypothesis is falsified.

**Expected outcome if true:** Kendall's τ_rank > 0.5 between first-letter and semantic-hierarchy architecture rankings.
**Expected outcome if false:** τ_rank ≤ 0 or statistically indistinguishable from 0.
