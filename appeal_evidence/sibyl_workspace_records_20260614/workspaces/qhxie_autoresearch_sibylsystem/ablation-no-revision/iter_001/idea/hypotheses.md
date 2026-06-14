# Testable Hypotheses: Limits of Consistency-Based Activation Energy

## Primary Hypotheses

### H1: Aggregate Arrhenius Saturation -- CONFIRMED

**Status**: CONFIRMED (pilot validation complete)

**Claim**: LLM reasoning accuracy follows exponential saturation at the aggregate level: P_k(correct) = P_inf * (1 - exp(-k/k_0))

**Test**: Sample k=1,2,4,8,16 per problem on MATH. Fit exponential saturation model.

**Evidence**:
- R^2 = 0.924 (threshold >0.85) on Qwen2.5-Math-7B-Instruct (n=50)
- P_inf = 0.835 (asymptotic ceiling)
- k_0 = 0.613 (characteristic sampling count)
- AIC/AICc/BIC all favor exponential over power-law and logarithmic models

**Caveats**:
- Per-problem median R^2 = 0.000; mean = 0.077
- 80% of individual problems cannot be fit (40/50 invalid k_0)
- Power-law wins 24/50 BIC comparisons at per-problem level

**Conclusion**: Arrhenius kinetics confirmed as aggregate statistical pattern, NOT as universal physical law.

---

### H2: Ea Correlates with Problem Difficulty -- CONFIRMED

**Status**: CONFIRMED (pilot validation complete)

**Claim**: Activation energy (Ea) estimated from answer consistency trajectory correlates with MATH level difficulty.

**Test**: Estimate Ea for each problem via consistency convergence rate. Compare Ea distribution across levels 1-5.

**Evidence**:
- Spearman(Ea, MATH level) = 0.448, p=0.001 (n=50)
- Pilot: Spearman = 0.578, p=0.0008 (n=30) -- consistent direction
- Level means: L1=9.47, L2=9.64, L3=9.75, L4=9.71, L5=10.0

**Caveats**:
- Effect size is modest (Spearman=0.448 = medium-weak)
- Ea values highly concentrated (bimodal at ~9.47 and ~10.0)
- Level 5 Ea numerically compressed (std ~1.9e-6) -- possible algorithmic artifact
- Level 3 and L4 means nearly identical (9.75 vs 9.71)
- Within-level discriminative power is minimal

**Conclusion**: Ea coarsely reflects problem difficulty (level-discriminative, not within-level).

---

### H3: Ea Predicts Single-Pass Solveability -- FALSIFIED

**Status**: FALSIFIED

**Original Claim**: Problems with Ea below threshold can be solved in single-pass with >75% accuracy.

**Evidence**:
- Low-Ea accuracy (full n=50): 75.0% -- but threshold was post-hoc optimized (data leakage)
- Low-Ea accuracy (pilot n=30): 68.4% < 75% -- FAIL
- AUC = 0.436 < 0.5 (decisive evidence of predictive failure)
- Spearman(Ea, accuracy) = -0.063, p=0.66 (zero correlation)
- Pearson(Ea, accuracy) = 0.024 (zero correlation)

**Falsification Reason**: Ea captures answer stability (consistency of responses) but NOT answer correctness. A problem can have low activation energy (consistent answers) but still be consistently wrong due to:
- Execution errors (calculation mistakes) that repeat across samples
- Conceptual gaps that produce stable wrong answers
- Answer extraction failures that appear consistent

**Conclusion**: Ea cannot be used as a routing signal for single-pass vs multi-sample decisions.

---

### H4: Ea Measures Stability, Not Correctness -- PENDING EMPIRICAL VALIDATION

**Claim**: The reason Ea fails for routing is that it captures answer agreement, not reasoning quality.

**Test**: Classify low-Ea failures into:
- Execution errors (calculation mistakes, algebra errors)
- Conceptual errors (wrong approach, misinterpretation)
- Answer extraction failures (pipeline errors)

**Expected**: Execution errors dominate low-Ea failures (>50%), supporting the "stable but wrong" narrative.

**Falsification**: If conceptual errors dominate (>50%), Ea is fundamentally limited in a different way.

**Time**: 30 minutes

---

### H5: Consistency-Ea and Saturation-k_0 Are Related -- FALSIFIED

**Status**: FALSIFIED

**Claim**: The two "activation energy" measures (from consistency and from saturation curve) capture the same construct.

**Evidence**:
- Spearman(Ea, k_0) = -0.219, p=0.543 (n=50, but only 10 valid pairs)
- Negative correlation direction (opposite of theoretical expectation)
- 80% of problems (40/50) lack valid per-problem k_0 estimates

**Interpretation**:
- Consistency-Ea measures **answer stability** (how quickly answers converge)
- Saturation-k_0 measures **model learning dynamics** (how quickly accuracy saturates)
- They are unrelated constructs

**Conclusion**: The theoretical unity of the "activation energy" framework is undermined. Two measures with the same name capture different phenomena.

---

## New Hypotheses (From Revisionist)

### H6: Ea Bimodality Reflects Two Reasoning Modes

**Claim**: The bimodal Ea distribution (~9.47 and ~10.0) reflects two distinct reasoning modes.

**Evidence**: Ea clusters at two values; Gaussian mixture model may confirm two components.

**Test**: Fit 2-component GMM to Ea distribution. Report BIC vs 1-component.

**Expected**: 2-component model fits significantly better.

---

### H7: Aggregate != Individual

**Claim**: Arrhenius model describes ensemble-average behavior, not individual problem behavior.

**Evidence**: Aggregate R^2=0.924 vs per-problem median R^2=0.000.

**Test**: Quantify deviation between aggregate curve and individual problem curves.

**Expected**: Large ensemble-average deviation for most problems.

---

### H8: Power-Law May Fit Individuals Better

**Claim**: Individual problems may follow power-law rather than exponential saturation.

**Evidence**: Power-law wins 24/50 BIC comparisons; mean per-problem R^2=0.220 (vs 0.077 for exponential).

**Test**: Formal model comparison on larger sample (n=250).

**Expected**: Power-law is preferred for >50% of individual problems.

---

## Null Hypotheses (Explicit Falsification Record)

| Hypothesis | Falsified? | Evidence |
|------------|------------|----------|
| H0-1: P_k does not follow exponential saturation (aggregate) | **REJECTED** | R^2 = 0.924 |
| H0-2: Ea does not correlate with difficulty | **REJECTED** | Spearman = 0.448, p=0.001 |
| H0-3: Ea predicts single-pass correctness | **CONFIRMED** | AUC = 0.436 < 0.5 |
| H0-5: Ea and k_0 are unrelated | **CONFIRMED** | Spearman = -0.219, p=0.54 |

---

## Pilot Evaluation Criteria

| Hypothesis | Metric | Threshold | Status |
|------------|--------|-----------|--------|
| H1 | Exponential fit R^2 (aggregate) | >0.85 | **CONFIRMED** (0.924) |
| H2 | Ea-level Spearman correlation | >0.4 | **CONFIRMED** (0.448) |
| H3 | Single-pass accuracy (low Ea) | >75% | **FALSIFIED** (68.4% pilot, AUC=0.436) |
| H4 | Execution error ratio | >50% | **PENDING** |
| H5 | Ea-k_0 Spearman correlation | >0.5 | **FALSIFIED** (-0.219) |

---

## Relationship to Prior Hypotheses

| Prior Round | Prior Hypothesis | Status | New Hypothesis |
|-------------|-----------------|--------|----------------|
| Round 1 | EDW-DPO depth weighting | FALSIFIED | N/A |
| Round 2 | DeepSeek >40% MATH | FALSIFIED (26%) | N/A |
| Round 2 | Step-DPO loss <0.5 | FALSIFIED (0.694) | N/A |
| Round 3 | API-based evaluation | BLOCKED | N/A |
| Round 4 | H1 Arrhenius kinetics | **CONFIRMED** (aggregate only) | H7, H8 |
| Round 4 | H2 Ea = difficulty | **CONFIRMED** (coarse) | H6 |
| Round 4 | H3 Single-pass threshold | **FALSIFIED** | H4 |
| Round 4 | H5 Ea = k_0 | **FALSIFIED** | N/A |

**Key difference**: Round 5 focuses on **diagnosing H3 failure** and **mapping theoretical boundaries**, not on finding alternative routing signals.

---

## Falsification Summary

| Hypothesis | Falsified If | Evidence Would Show |
|------------|--------------|---------------------|
| H1 (Arrhenius) | Power-law fits better at aggregate | P_k follows P_k = k^(-alpha) -- **REJECTED** |
| H2 (Ea = difficulty) | No correlation | Ea randomly distributed -- **REJECTED** |
| H3 (Single-pass threshold) | Low-Ea accuracy < 75% OR AUC < 0.5 | **CONFIRMED FALSIFICATION** (AUC=0.436) |
| H4 (Execution errors) | Conceptual errors dominate | >50% conceptual errors -- PENDING |
| H5 (Ea = k_0) | No positive correlation | Spearman <= 0 -- **CONFIRMED FALSIFICATION** (-0.219) |
| H6 (Bimodality) | Single Gaussian fits better | GMM BIC > single Gaussian -- PENDING |
| H7 (Aggregate != Individual) | Small deviation | Per-problem curves close to aggregate -- PENDING |
| H8 (Power-law individuals) | Exponential wins most | Exponential BIC > power-law for >50% -- PENDING |
