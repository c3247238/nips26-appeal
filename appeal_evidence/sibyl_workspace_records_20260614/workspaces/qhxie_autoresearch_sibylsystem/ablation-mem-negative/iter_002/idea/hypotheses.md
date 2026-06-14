# Research Hypotheses: Iteration 2 — UAD Generalization with Dead Feature Analysis

## Primary Hypotheses

### H1: Dead Feature Confound on UAD
**Statement**: High dead feature ratios (>90%) reduce UAD precision by shrinking the effective dictionary size, but UAD recall remains >= 0.9 because absorption signatures (high conditional probability, low marginal frequency) persist even in sparse dictionaries.

**Falsification Criteria**:
- If recall < 0.7 on any SAE with dead feature ratio > 80%, H1 is falsified.
- If precision does NOT improve when dead features are filtered out before clustering, H1 is falsified.

**Measurement**:
- Precision and recall computed with and without dead feature filtering
- Effective dictionary size = d_SAE * (1 - dead_feature_ratio)
- Correlation between dead_feature_ratio and UAD precision

**Control**: Same UAD algorithm with fixed hyperparameters; only dead feature filtering varies.

---

### H2: Cross-Architecture Collision Convergence
**Statement**: When comparing SAEs with similar dead feature ratios, collision rates converge across architectures (JumpReLU vs TopK), confirming the Iteration 1 architecture difference was confounded by dead features.

**Falsification Criteria**:
- If collision rate difference > 0.05 between architectures with matched dead feature ratios, H2 is falsified.
- If collision rate correlates more strongly with architecture type than with dead feature ratio, H2 is falsified.

**Measurement**:
- Collision rate per architecture
- Dead feature ratio per SAE
- Partial correlation controlling for dead feature ratio

**Control**: Compare against Iteration 1's unmatched comparison (15.4% vs 3.8% with 94% vs 96% dead features).

---

### H3: UAD Generalization to Healthier Dictionaries
**Statement**: UAD achieves F1 >= 0.55 on pretrained SAEs with dead feature ratios < 50%, confirming generalization beyond the high-dead-feature regime.

**Falsification Criteria**:
- If F1 < 0.5 on any pretrained SAE with dead feature ratio < 50%, H3 is falsified.
- If F1 variance across SAEs correlates more strongly with model size than with dead feature ratio, H3 is falsified.

**Measurement**:
- F1 on each pretrained SAE
- Dead feature ratio per SAE
- Cross-SAE consistency of detected pairs

**Control**: Random feature pair selection baseline.

---

### H4: DFDA on Pretrained Absorbed Pairs
**Statement**: DFDA achieves >=10% MSE improvement on absorbed pairs identified in pretrained SAEs, confirming compensation works on healthier dictionaries.

**Falsification Criteria**:
- If mean improvement < 5%, H4 is falsified.
- If fewer than 60% of pairs show positive improvement, H4 is falsified.

**Measurement**:
- MSE improvement per pair
- Reconstruction MSE change
- Parameter count ratio

**Control**: No-compensation baseline.

---

## Supplementary Hypotheses

### H-S1: Effective Dictionary Size vs Collision Rate
**Statement**: Collision rate increases as effective dictionary size decreases, following a species-area-like relationship (interdisciplinary hypothesis).

**Measurement**: Log-log regression of collision_rate vs effective_dictionary_size.

### H-S2: UAD Precision Decomposition
**Statement**: UAD false positives are predominantly (a) correlated but non-absorbed features and (b) features affected by dead feature sparsity, not random errors.

**Measurement**: Manual inspection of false positive pairs; classification into error types.

---

## Hypothesis Dependency Graph

```
H1 (dead feature confound on UAD)
  |
  +---> H2 (cross-architecture convergence) [independent validation]
  |
  +---> H3 (UAD on healthier dictionaries) [depends on H1 for interpretation]
          |
          +---> H4 (DFDA on pretrained) [depends on H3 for pair identification]
```

## Go/No-Go Gates

| Gate | Condition | If Passed | If Failed |
|------|-----------|-----------|-----------|
| G1 | H3: F1 >= 0.55 on >=1 pretrained SAE | Proceed to H4 | If F1 >= 0.5, proceed with caveat; if < 0.5, PIVOT |
| G2 | H4: Mean improvement >= 10% on >=4 pairs | Paper retains UAD+DFDA | Downgrade DFDA; paper focuses on UAD + dead feature analysis |
| G3 | H1: Precision improves with dead feature filtering | Include dead feature analysis as contribution | Report as negative result; UAD robustness is the story |

## Pre-registered Analysis Plan

1. **Primary Analysis**: Report mean F1 / MSE improvement with bootstrap 95% CI.
2. **Multiple Comparisons**: Bonferroni correction when testing across multiple SAEs (alpha = 0.05 / n_SAEs).
3. **Negative Result Handling**: Report all falsified hypotheses prominently.
4. **Power Analysis**: Report achieved power; note if < 0.8.
