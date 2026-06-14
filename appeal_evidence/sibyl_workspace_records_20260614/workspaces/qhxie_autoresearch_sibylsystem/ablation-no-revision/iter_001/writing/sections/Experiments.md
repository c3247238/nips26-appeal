# 4. Experiments

We present empirical validation of Activation Energy Theory using *Qwen2.5-Math-7B-Instruct* on the MATH dataset. Our pilot experiments test three hypotheses: (H1) accuracy follows exponential saturation, (H2) activation energy correlates with problem difficulty, and (H3) low-$E_a$ problems can be solved in single-pass with >75% accuracy.

## 4.1 Experimental Setup

### 4.1.1 Model and Dataset

We evaluate on *Qwen2.5-Math-7B-Instruct*, a 7-billion parameter instruction-tuned model specialized for mathematics. The model was loaded in bfloat16 precision on a single RTX PRO 6000 Blackwell GPU (sm_120) via PyTorch 2.11.0.

We use the MATH test set, a benchmark of 12,500 mathematics competition problems across five difficulty levels [1]. For pilot experiments, we use a subset of 100 problems for main analysis and 30 problems for per-group hypothesis testing, selected to ensure coverage across all five difficulty levels.

### 4.1.2 Evaluation Protocol

For each problem $x_i \in \mathcal{D}$, we generate $k \in \{1, 2, 4, 8, 16\}$ independent samples using temperature $T = 0.7$ with maximum 1024 tokens. The majority-voted answer is compared against the ground truth $y_i^*$ to compute accuracy.

We measure three key quantities per problem:
- **Accuracy@$k$**: Whether the majority answer matches $y_i^*$ after $k$ samples
- **Answer consistency $c_{i,k}$**: Fraction of samples agreeing on the majority answer
- **Token counts**: Total tokens generated for cost estimation

### 4.1.3 Pilot Configuration

Table 1 summarizes our experimental configuration.

**Table 1: Experiment Configuration**

| Parameter | Value |
|-----------|-------|
| Model | *Qwen2.5-Math-7B-Instruct* |
| Dataset | MATH test set (subset) |
| Temperature | 0.7 |
| Max tokens | 1024 |
| Sample counts $k$ | 1, 2, 4, 8, 16 |
| Main sample size $n$ | 100 |
| Analysis sample size $n$ | 30 |
| Timeout | 900s per group |
| Random seed | 42 |

## 4.2 Results

### 4.2.1 H1: Arrhenius Kinetics Validation

**Finding: Exponential saturation confirmed with $R^2 = 0.936$**

We test whether accuracy follows the Arrhenius kinetics model $P_k = P_\infty \cdot (1 - e^{-k/k_0})$. Table 2 shows accuracy at each sampling count.

**Table 2: Accuracy vs. Sampling Count (n=30)**

| $k$ | Accuracy |
|-----|----------|
| 1 | 66.7% |
| 2 | 76.7% |
| 4 | 80.0% |
| 8 | 83.3% |
| 16 | 83.3% |

We fit the exponential saturation model using nonlinear least squares. The fitted parameters are:
- **Asymptotic ceiling $P_\infty = 0.818$**: The model's maximum achievable accuracy regardless of sample count
- **Characteristic count $k_0 = 0.613$**: The rate at which accuracy saturates with additional samples
- **$R^2 = 0.936$**: The model explains 93.6% of variance in accuracy across $k$ values

The $R^2$ exceeds our threshold of 0.85, confirming H1. The characteristic count $k_0 = 0.613$ indicates that most accuracy improvement occurs in the first few samples; going from $k=1$ to $k=2$ yields +10 percentage points while $k=8$ to $k=16$ yields +0 points.

Figure 3 visualizes the saturation curve with the fitted exponential model.

<!-- FIGURES
- Figure 3: gen_h1_saturation.py, h1_saturation.pdf — Accuracy vs. k with exponential fit (R²=0.936)
-->

### 4.2.2 H2: Activation Energy Correlates with Difficulty

**Finding: Activation energy correlates with MATH level (Spearman $\rho = 0.578$, $p = 0.0008$)**

We estimate activation energy $\hat{E}_a$ from the consistency convergence rate: problems where samples quickly agree on an answer (high initial consistency) have lower $\hat{E}_a$ (easier), while problems with slow consistency convergence have higher $\hat{E}_a$ (harder).

We compute Spearman correlation between $\hat{E}_a$ and the MATH difficulty level $L_i \in \{1, 2, 3, 4, 5\}$. Table 3 shows mean $\hat{E}_a$ by level.

**Table 3: Mean Activation Energy by MATH Level (n=30)**

| MATH Level | Mean $\hat{E}_a$ | Sample Size |
|------------|------------------|-------------|
| 1 | 9.47 | 4 |
| 2 | 9.60 | 9 |
| 3 | 9.54 | 7 |
| 4 | 9.67 | 6 |
| 5 | 9.94 | 4 |

The Spearman correlation $\rho = 0.578$ exceeds our threshold of 0.4 (/$p = 0.0008$). Higher MATH levels correspond to higher activation energy, confirming H2.

Figure 4 shows the distribution of $\hat{E}_a$ across difficulty levels with individual data points.

<!-- FIGURES
- Figure 4: gen_h2_ea_difficulty.py, h2_ea_difficulty.pdf — Ea distribution across MATH difficulty levels
-->

### 4.2.3 H3: Single-Pass Threshold

**Finding: FALSIFIED — Ea does not predict single-pass solveability**

We test whether problems with low activation energy can be solved in a single pass with >75% accuracy. We split problems into low-$E_a$ (bottom 50%) and high-$E_a$ (top 50%) groups and measure single-pass accuracy for each.

**Table 4: Single-Pass Accuracy by Activation Energy Category (n=30)**

| Category | Single-Pass Accuracy | Threshold |
|----------|---------------------|-----------|
| Low-$E_a$ | 68.4% | 75% |
| High-$E_a$ | 63.6% | — |

The low-$E_a$ group achieves 68.4% accuracy, falling short of our 75% threshold. This **falsifies H3**. Counterintuitively, the high-$E_a$ group also achieves only 63.6%, suggesting that activation energy from answer consistency does not predict single-pass solveability.

Figure 5 compares single-pass accuracy against the threshold, illustrating the routing failure.

**Why does Ea fail?** Activation energy estimated from answer consistency measures **agreement** among samples, not **correctness**. A problem can have low activation energy (all samples agree on an answer) but still be wrong if the samples make the same execution errors or share the same conceptual mistake.

<!-- FIGURES
- Figure 5: gen_h3_routing.py, h3_routing.pdf — Single-pass accuracy by Ea category vs. 75% threshold
-->

### 4.2.4 Hypothesis Summary

Table 5 summarizes all three hypotheses and their outcomes.

**Table 5: Hypothesis Results Summary**

| Hypothesis | Metric | Threshold | Actual | Status |
|------------|--------|-----------|--------|--------|
| H1: Arrhenius kinetics | $R^2$ | > 0.85 | 0.936 | **CONFIRMED** |
| H2: $E_a$ = difficulty | Spearman $\rho$ | > 0.4 | 0.578 | **CONFIRMED** |
| H3: Single-pass threshold | Accuracy (low $E_a$) | > 75% | 68.4% | **FALSIFIED** |

<!-- FIGURES
- Figure 6: gen_hypothesis_summary.py, hypothesis_summary.pdf — Summary visualization of H1-H3 results
-->

## 4.3 Discussion of Findings

### 4.3.1 Implications of Confirmed Hypotheses

H1 and H2 provide empirical support for the Activation Energy Theory framework. The exponential saturation model ($R^2 = 0.936$) accurately describes how accuracy improves with sampling count, and the correlation between activation energy and problem difficulty ($\rho = 0.578$) confirms that consistency signals capture difficulty information.

However, these findings do not enable reliable routing decisions. The falsification of H3 reveals a fundamental limitation: difficulty and solveability are distinct concepts.

### 4.3.2 The Difficulty-Solveability Gap

Our negative result exposes a critical gap in consistency-based routing. Activation energy measures how consistently a model answers a problem, but it does not measure whether those answers are correct. This gap arises from three failure modes in low-$E_a$ problems:

1. **Execution errors**: All samples make the same arithmetic mistake (e.g., miscalculating $7 \times 8 = 54$), producing high consistency but incorrect answers
2. **Conceptual errors**: All samples use the wrong approach (e.g., solving a geometry problem algebraically), producing internally consistent but wrong reasoning
3. **Answer extraction failures**: Samples correctly solve the problem but fail to extract the answer in the expected format

Figure 6 conceptualizes this gap by comparing consistency-based routing signals against alternative approaches.

<!-- FIGURES
- None
-->

## 4.4 Limitations

Our pilot experiments have several limitations:

1. **Sample size**: 30 problems per hypothesis test provides limited statistical power; smaller effects may be undetected
2. **Single model**: *Qwen2.5-Math-7B-Instruct* may not generalize to other models or domains
3. **Synthetic activation energy**: We estimate $E_a$ from consistency convergence rather than fitting $k_0$ directly for each problem
4. **Self-fulfilling threshold**: The 75% threshold was chosen a priori; different thresholds may yield different conclusions

---

## References

[1] Hendrycks, D., et al. (2021). Measuring Mathematical Problem Solving with the MATH Dataset. *NeurIPS Datasets Track*.
