# Critique of Ideation

## Overall Framing

The research question is well-motivated: Is feature absorption a failure mode that degrades SAE-based interpretability, or is it a benign structural artifact? This is an important question for the field.

## Strengths

1. **Multiple hypothesis approach**: The project systematically tests multiple hypotheses (H1-H10), which is methodologically sound.

2. **Random baseline comparison**: The H7/H10 random SAE baseline comparison is a genuinely novel contribution that other work has not done. This is the strongest empirical finding.

3. **Multiple comparison correction**: Applying Bonferroni and BH-FDR correction is methodologically appropriate and demonstrates rigor.

4. **Honest null-result reporting**: The paper does report null results without spin in the methods and results sections.

## Weaknesses

### 1. Post-Hoc Pivot to "Optimal Compression"

The "optimal compression" framing emerged after the experiments failed to find significant effects. This is a post-hoc rationalization rather than a pre-registered hypothesis. The paper should either:

- Remove the optimal compression framing and present the paper as a null-result study
- Pre-register the compression hypothesis and run experiments to test it directly

### 2. The Mechanistic Story is Unsupported

The paper connects absorption to LCA (Locally Competitive Algorithm) and decoder correlations, claiming this explains absorption as "competitive suppression." However:

- H6 (decoder graph predicts absorption pairs) is **falsified** (precision@20 = 0.0)
- H8 (graph statistics predict at-risk features) is **not supported** (r=0.12, p=0.55)

The mechanistic story is used to explain why null results should be interpreted positively, but the mechanism itself is not empirically validated.

### 3. Underpowered Study

With n=26 features, the study has approximately 20% power to detect a medium effect size (r=0.5). This means:
- 80% chance of failing to detect a real effect
- Null results are uninterpretable

The paper should acknowledge this prominently rather than presenting null results as positive findings.

### 4. The Metric Itself is Questioned But Used

The paper notes that the Chanin absorption metric shows high absorption (0.278) in random SAEs, suggesting the metric may be flawed. However, the paper then uses this same flawed metric to support its conclusions. This is circular reasoning.

### 5. Alternative Interpretations

The findings are equally consistent with multiple interpretations:

**Interpretation A (what the paper argues):** Absorption is benign and training reduces structural artifacts.

**Interpretation B:** The experiments are underpowered and failed to detect real effects.

**Interpretation C:** The absorption metric is flawed and measures dictionary structure rather than learned pathology.

**Interpretation D:** Absorption affects downstream tasks but in ways not captured by steering and probing metrics.

The paper does not adequately discuss alternatives B, C, and D.

## Recommendations for Ideation

1. **Pre-register hypotheses**: Before running experiments, clearly state what would constitute evidence for vs. against the "absorption is benign" hypothesis.

2. **Power analysis**: Conduct a-priori power analysis to determine required sample size. If n=26 is insufficient, either increase sample size or reframe the study as exploratory.

3. **Direct tests**: Test the optimal compression prediction directly rather than inferring it from null results.

4. **Acknowledge alternative interpretations**: The discussion should explicitly consider alternative explanations for the findings.
