# Pilot Experiment Summary

## Overall Recommendation: GO

### Task: p1_multichild_absorption (PILOT)

## Pilot Results

### H1: Multi-child Proportional Absorption

| Condition | Absorption Rate | Std | vs Trained SAE |
|-----------|----------------|-----|----------------|
| Trained SAE | 0.5000 | 0.0000 | --- |
| Random Decoder | 0.0590 | 0.0694 | -0.4410 |
| Shuffled Features | 0.4870 | 0.0336 | -0.0130 |
| Permuted Encoder | 0.4840 | 0.0367 | -0.0160 |

**Statistical Tests (Trained SAE vs Baselines):**

| Comparison | t-statistic | p-value | Cohen's d |
|------------|-------------|---------|-----------|
| vs Random Decoder | 63.209 | 3.16e-133 | 8.939 |
| vs Shuffled Features | 3.846 | 1.62e-04 | 0.544 |
| vs Permuted Encoder | 4.342 | 2.25e-05 | 0.614 |

**H1 Pass Criteria:**
- Trained SAE absorption > 0.3: **PASS** (0.5000)
- Trained SAE > Random baseline: **PASS** (delta = 0.4410)
- p-value < 0.05: **PASS** (p = 3.16e-133)

**OVERALL H1: PASS**

### H2: Frequency-Absorption Correlation

| Metric | Value |
|--------|-------|
| Spearman rho | 0.1711 |
| p-value | 6.47e-08 |
| Pearson r | 0.3631 |
| p-value | 4.21e-32 |

**H2 Pass Criteria:**
- rho < -0.3: **FAIL** (rho = 0.1711, positive correlation)
- p < 0.01: **PASS**

**OVERALL H2: FAIL**

### Proportional Variance

| Condition | Mean | Std |
|-----------|------|-----|
| Trained SAE | 0.1154 | 0.0072 |
| Random Decoder | 0.0040 | 0.0011 |
| Shuffled Features | 0.1057 | 0.0281 |
| Permuted Encoder | 0.1031 | 0.0304 |

## Key Findings

### H1 Supported (Strong Evidence)
1. **Trained SAEs show significantly higher absorption (0.50) than random baselines (0.059)**
   - Very large effect size (Cohen's d = 8.94)
   - p-value < 10^-133 (extremely significant)
2. **Shuffled and permuted baselines show intermediate absorption (~0.48)**
   - Slightly lower than trained SAE but much higher than random decoder
   - Suggests encoder weight structure contributes to absorption patterns

### H2 Not Supported (Negative Result)
1. **Absorption does NOT inversely correlate with feature frequency**
   - Actually shows positive correlation (rho = 0.17)
   - Higher-frequency features tend to have higher absorption
   - Contradicts competitive exclusion hypothesis

### Methodological Insights
1. **Multi-child proportional absorption successfully differentiates trained from random**
   - Uses overlap between parent features and each child SEPARATELY
   - Average of parent-child1 and parent-child2 overlap
2. **Single-child ablation saturates (pilot finding)**
   - Multi-child approach is necessary for differentiation
3. **Proportional variance shows trained SAE has more asymmetric child contributions**

## Recommendations

1. **H1 is ready for full experiment**: Very strong evidence supporting the hypothesis
2. **H2 needs revision**: The frequency-correlation hypothesis is not supported; consider dropping or reformulating
3. **Consider ablation of H2**: Replace with alternative H2 or document as negative result
4. **Proceed to full experiment**: Run with 5 seeds × 3 L0 targets × 4 conditions

## Next Steps

1. Implement full experiment (5 seeds, L0 ∈ {16, 32, 64})
2. Implement H3 steering intervention (if time permits)
3. Implement H_Safe safety-critical analysis (if time permits)
4. Generate figures for paper
