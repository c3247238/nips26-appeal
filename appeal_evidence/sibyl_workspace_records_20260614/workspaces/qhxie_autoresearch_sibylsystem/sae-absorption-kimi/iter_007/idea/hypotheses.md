# Testable Hypotheses with Expected Outcomes

## H1: L0-Matched Architecture Effects (RQ1)

### H1a: Matryoshka maintains lower absorption at matched L0
- **Directional prediction**: Matryoshka absorption rate < Baseline absorption rate at same L0
- **Expected effect size**: Cohen's d > 1.0 (large effect, replicating Bussmann et al. claim)
- **Expected values**: Matryoshka ~0.05, Baseline ~0.49 (at L0=50)
- **Test**: Welch's t-test, 5 seeds per variant
- **Falsification threshold**: If d < 0.5 or p > 0.05, claim is not supported under our conditions

### H1b: OrtSAE shows no benefit over L0-matched Baseline (untuned lambda)
- **Directional prediction**: OrtSAE absorption rate == Baseline absorption rate at matched L0
- **Expected effect size**: Cohen's d < 0.2 (negligible)
- **Rationale**: Iter 006 found null result; lambda_ortho was not tuned
- **Test**: Welch's t-test
- **Falsification threshold**: If d > 0.8, prior null result was specific to our hyperparameters

### H1c: TopK shows higher absorption than L0-matched Baseline
- **Directional prediction**: TopK absorption rate > Baseline absorption rate at same L0
- **Expected effect size**: Cohen's d > 0.8 (consistent with SAEBench finding)
- **Rationale**: TopK optimizes reconstruction, which trades off against absorption
- **Test**: Welch's t-test
- **Falsification threshold**: If d < 0.2, SAEBench finding may be specific to their evaluation protocol

---

## H2: Absorption-Downstream Causality (RQ2)

### H2a: Absorption negatively correlates with sparse probing F1
- **Directional prediction**: Pearson r < -0.5 between absorption rate and sparse probing F1
- **Expected trend**: Linear or sublinear decrease in F1 as absorption increases
- **Test**: Pearson correlation on individual replicates (n >= 20)
- **Falsification threshold**: If r > -0.3 or p > 0.05, no reliable correlation

### H2b: Absorption negatively correlates with steering efficacy
- **Directional prediction**: Pearson r < -0.5 between absorption rate and steering effect size
- **Expected mechanism**: Absorbed parent features cannot be independently steered
- **Test**: Pearson correlation
- **Falsification threshold**: If r > -0.3, absorption does not impair steering

### H2c: Causality via sparsity manipulation
- **Directional prediction**: Increasing lambda_L1 (increasing absorption) degrades downstream metrics
- **Expected trend**: Monotonic degradation with increasing sparsity/absorption
- **Test**: Linear regression of downstream metric on absorption rate, controlling for architecture
- **Falsification threshold**: If slope is not significantly negative (p > 0.05), correlation is not causal

---

## H3: Mutual Coherence Predictor (RQ3)

### H3a: Mutual coherence positively correlates with absorption
- **Directional prediction**: Pearson r > 0.5 between mu(W_dec) and absorption rate
- **Expected mechanism**: Higher decoder coherence enables feature sharing -> absorption
- **Test**: Pearson correlation across all variants and seeds
- **Falsification threshold**: If r < 0.3, decoder coherence is not a useful absorption predictor

### H3b: Theoretical threshold predicts absorption onset
- **Directional prediction**: Absorption rate increases sharply when mu > 1/(2k-1)
- **Expected pattern**: Sigmoid-like relationship with threshold at mu = 1/(2k-1)
- **Test**: Logistic regression or piecewise linear model with breakpoint at 1/(2k-1)
- **Falsification threshold**: If AUC < 0.7 for threshold-based classification, theory does not empirically hold

---

## H4: Task Generalization (RQ4)

### H4a: First-letter absorption correlates with semantic absorption
- **Directional prediction**: Pearson r > 0.5 between first-letter absorption rate and semantic absorption rate
- **Expected implication**: First-letter tasks are valid proxies for real features
- **Test**: Pearson correlation across architectures
- **Falsification threshold**: If r < 0.3, first-letter metrics are unrepresentative

### H4b: Semantic absorption rates differ by feature category
- **Directional prediction**: Absorption rates differ significantly across syntactic/factual/safety features
- **Expected pattern**: Safety features may show higher absorption (more abstract parent concepts)
- **Test**: One-way ANOVA across feature categories
- **Falsification threshold**: If F-test p > 0.05, absorption is uniform across semantic domains

---

## Pre-registered Analysis Plan

### Primary Analyses (must be reported regardless of outcome)

1. **H1a-c**: Welch's t-test for each architecture vs. L0-matched Baseline
2. **H2a-b**: Pearson correlation with 95% CI between absorption and downstream metrics
3. **H3a**: Pearson correlation between mutual coherence and absorption

### Secondary Analyses (report if significant, note if not)

4. **H2c**: Linear regression of downstream metrics on absorption, controlling for architecture
5. **H3b**: Threshold model fit for mu = 1/(2k-1)
6. **H4a-b**: Cross-task correlation and ANOVA

### Exploratory Analyses (report with appropriate caveats)

7. Interaction effects: Does architecture moderate the absorption-downstream relationship?
8. Layer-wise patterns: Does absorption vary across model layers?
9. Dead latent confounding: Does dead latent rate correlate with absorption?

---

## Decision Rules for Pivot

| Hypothesis outcome | Decision |
|-------------------|----------|
| H2a AND H2b supported (r < -0.5, p < 0.05) | PROCEED with front-runner; consider Alternative A for next iteration |
| H2a OR H2b null (r > -0.3 or p > 0.05) | PIVOT to Alternative D (absorption as compositional semantics) |
| H3a AND H3b supported | PROCEED; consider Alternative C (architecture design) |
| H3a supported but H3b null | Report as partial validation; theory needs refinement |
| H4a null (first-letter doesn't generalize) | Flag as major methodological concern; prioritize semantic evaluation |
| All H1-H4 null or weak | PIVOT to Alternative D; absorption may not be the right framing |
