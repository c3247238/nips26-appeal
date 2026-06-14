# Experiment Critique: Feature Absorption Study

## Summary

The experiments are well-designed with appropriate controls and statistical tests. However, several critical inconsistencies between reported numbers and the paper's claims suggest either data integrity issues or inadequate verification before reporting.

## Critical Issues

### Issue 1: H_Mech Decomposition is Internally Inconsistent

**Claim in paper**: "decoder geometry contributes 0.299 while encoder alignment adds 0.185"

**Table 5 data**:
- Condition A (Random/Random): 0.299
- Condition C (Random/Trained): 0.299

If Condition C = Condition A, then trained decoder adds NOTHING beyond random decoder geometry. The decoder contribution is 0, not 0.299.

The decomposition logic appears flawed:
- Geometric contribution should be: abs(C) - abs(A) = 0.299 - 0.299 = 0
- Learned contribution would be: abs(D) - abs(C) = 0.484 - 0.299 = 0.185

But the paper claims geometric contribution = 0.299. This contradicts the table data.

**This is a qualitative failure, not a sensitivity issue.**

### Issue 2: Shuffled/Permuted Baselines Exceed Condition C

**Table 1**:
- Random Decoder: 0.059
- Shuffled Features: 0.487
- Permuted Encoder: 0.484

Shuffled/permuted baselines (which preserve geometry but break feature identity) achieve 0.487-0.484, nearly the same as trained SAE (0.500).

But Condition C (decoder-only trained) achieves only 0.299.

This suggests:
1. Decoder geometry alone produces 0.299 (from H_Mech)
2. Breaking feature identity (shuffled) jumps to 0.487

What produces the additional 0.188 absorption in shuffled/permuted? The paper doesn't explain this gap.

### Issue 3: H_Safe Uses Different Methodology

Section 3.7 says H_Safe measures absorption "via the overlap method" but the paper never defines this method. All other experiments use multi-child proportional ablation.

This methodology switching is unexplained. If multi-child proportional ablation is the gold standard, why not use it for H_Safe?

### Issue 4: Table 1 Zero Standard Deviation

Trained SAE shows absorption rate 0.5000 with Std=0.0000.

Across 5 seeds, it's suspicious that the value is exactly 0.5000 with no variance. This suggests either:
- The value is exact (e.g., 0.5 by construction)
- It's a single measurement reported incorrectly

The paper should clarify this.

## Design Quality

### Strengths

1. **Appropriate baselines**: Random decoder, shuffled features, permuted encoder test three distinct null hypotheses
2. **Factorial design**: 2x2 decomposition is the right approach for mechanism isolation
3. **Statistical rigor**: Large effect sizes, proper tests reported
4. **Synthetic ground truth**: Enables precise measurement

### Weaknesses

1. **Single architecture tested**: Only TopK on d=512 synthetic data. JumpReLU, gated SAEs, real transformer activations not tested.
2. **Sample size for H_Safe**: n=20 per group provides limited power for detecting medium effects
3. **No pre-registration**: Sample sizes (n=20 for H3) appear post-hoc, not pre-registered

## Statistical Concerns

### No p-value for B>D Comparison

The paper highlights B>D as a central anomaly but provides no statistical test comparing B vs D directly. With a difference of only 0.006, this could easily be noise.

### Non-Monotonic Steering Percentages

Table 6 shows:
- alpha=0.5: 134,717,856%
- alpha=1.0: 472%

This non-monotonic pattern (decreasing percentage as alpha increases) indicates the computation is unstable, not a meaningful metric.

### H_Safe Null Result Interpretation

p=0.665 with n=20 per group means the study has limited power. The null result could mean:
1. No true difference exists
2. The study was underpowered to detect a real difference

The paper should acknowledge the power limitation when interpreting the null result.

## Missing Experiments

1. **H_Downstream**: Never conducted despite being the highest-value practical question
2. **H_Comp**: Listed in proposal but absent from paper
3. **H_Pareto**: Suspended due to formula bugs

## Falsification Criteria Review

| Hypothesis | Pre-registered Criterion | Actual Result | Assessment |
|------------|-------------------------|---------------|------------|
| H1 | p < 0.05, delta > 0.15 | d=8.94, delta=0.441 | PASSED |
| H2 | rho < -0.3 | rho=+0.171 | CORRECTLY FALSIFIED |
| H_Mech | Condition C ≈ D | C=0.299, D=0.484 | FALSIFIED |
| H3 | Absorbed > Non-absorbed sensitivity | 1.62x ratio | PASSED |
| H_Safe | p < 0.05 | p=0.665 | CORRECTLY FALSIFIED |

**Note**: H_Mech's falsification criterion was "geometric > learned contribution". With geometric=0 and learned=0.185, geometric is NOT greater. H_Mech should be marked as FALSIFIED or partially supported, not "SUPPORTED with nuance."

## Recommendations

1. **Fix the decomposition logic** - either the table data or the interpretation is wrong
2. **Explain shuffled/permuted gap** - the 0.43 jump from random decoder to shuffled needs explanation
3. **Define "overlap method"** or align H_Safe with multi-child methodology
4. **Add B vs D statistical test** or don't highlight B>D as an anomaly
5. **Clarify Table 1 Std=0** - if truly constant, say so