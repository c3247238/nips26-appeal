# Critique of Experiment Design and Execution

## Statistical Analysis

### Critical Issue: Multiple Comparisons

The paper conducts 12 hypothesis tests with zero significant results after proper multiple comparison correction:

| Test | Layer | r | uncorrected p | Bonferroni p | BH-FDR q |
|------|-------|---|---------------|--------------|----------|
| H1 raw steering | L4 | +0.008 | 0.970 | 1.0 | 1.0 |
| H1 raw steering | L8 | -0.301 | 0.136 | 1.0 | 0.544 |
| H1b delta steering | L4 | +0.245 | 0.227 | 1.0 | 0.681 |
| H1b delta steering | L8 | -0.431 | 0.028 | 0.334 | 0.107 |
| H2 probing | L4 | -0.003 | 0.987 | 1.0 | 1.0 |
| H2 probing | L8 | -0.107 | 0.604 | 1.0 | 1.0 |

**The paper presents H1b at L8 (p=0.028) as a key finding, but this does not survive correction.** The abstract, introduction, and conclusion all use this uncorrected p-value as evidence of a real effect, which is methodologically incorrect.

### Critical Issue: Statistical Power

With n=26 features:
- 20% power to detect medium effect (r=0.5)
- 80% probability of false negative

This makes null results uninterpretable. The paper acknowledges this in Section 3.6 but frames it as a post-hoc power analysis, which is methodologically questionable.

### Issue: CMI Dimension Instability

The CMI-absorption correlation reverses sign across probing dimensions:
- d'=10: rho=-0.383 (p=0.059)
- d'=20: rho=+0.048
- d'=30: rho=+0.299
- d'=50: rho=+0.197

This qualitative instability across measurement conditions suggests the phenomenon is not robust.

### Issue: Probe Quality Confound

Absorption rate correlates with probe F1 at rho=-0.67 (p<0.001). The paper does not control for probe quality in the CMI-absorption analysis. Features with low probe quality may show both low CMI (poor measurement) and high absorption (measurement artifact).

## Experimental Design

### Issue: Sample Size

n=26 first-letter features is very small for a correlation study. The first-letter hierarchy is also shallow (letter-level categories), which may not capture the semantic hierarchies where absorption is most relevant.

### Issue: Single Model

All experiments use GPT-2 Small. The paper acknowledges this limitation. Cross-model validation on Pythia-70m was inconclusive (p=0.841 for both raw and delta correlations).

### Issue: Single SAE Architecture

Only gpt2-small-res-jb SAE tested. Other SAE architectures (JumpReLU, Gated, TopK) may show different absorption patterns.

### Issue: Steering as the Primary Metric

The paper relies heavily on steering effectiveness as the primary downstream task. However:
- Steering bypasses the encoder entirely (adds decoder direction directly to residual stream)
- Steering robustness is therefore expected regardless of absorption
- Probing is a better test of encoder functionality

## Experiment Execution

### Positive: Rigorous Controls

- Random steering baseline for delta correction
- Random SAE baseline for metric validation
- Multiple layers tested (L0, L4, L8, L10)
- Multiple steering strengths tested

### Issue: H6 Falsification Not Adequately Addressed

H6 (decoder graph predicts absorption pairs) is falsified decisively:
- Precision@20 = 0.0
- Fisher exact test p = 1.0
- No enrichment over chance

Yet the paper continues to use the LCA/decoder correlation framework as the theoretical backbone.

### Issue: Homeostatic Rebalancing Not Tested

H10 (homeostatic rebalancing) was deferred but is presented as a contribution. Without empirical validation, this is speculative.

## Recommendations

1. **Prominently acknowledge power limitation**: The study is severely underpowered. Null results at this power level are not informative.

2. **Remove claims based on uncorrected p-values**: H1b at L8 should not be presented as evidence of a real effect.

3. **Control for probe quality**: Compute partial correlation of CMI vs. absorption controlling for probe F1.

4. **Address CMI dimension instability**: Either restrict analysis to d'=10 (pre-registered) or acknowledge the instability.

5. **Increase sample size**: Use more features or aggregate across multiple feature types.

6. **Test homeostatic rebalancing empirically**: Do not present it as a contribution without validation.
