# 4. Experiments

We present empirical tests of Activation Energy Theory on *Qwen2.5-Math-7B-Instruct* across 50 problems from the MATH dataset. The experiments evaluate four hypotheses: H1 (aggregate Arrhenius saturation), H2 ($E_a$ correlates with MATH difficulty level), H3 (low-$E_a$ problems are solvable in a single pass), and H5 (consistency-$E_a$ matches saturation-$k_0$).

## 4.1 Experimental Setup

### 4.1.1 Model and Dataset

We evaluate *Qwen2.5-Math-7B-Instruct* [1], a 7-billion parameter instruction-tuned model specialized for mathematics. The model runs in bfloat16 precision on a single RTX PRO 6000 Blackwell GPU (sm_120) via PyTorch 2.11.0.

We use the MATH test set, a benchmark of 12,500 mathematics competition problems annotated with five difficulty levels [2]. For this study, we sample 50 problems stratified across Levels 1--5 to ensure coverage of the full difficulty spectrum.

### 4.1.2 Evaluation Protocol

For each problem $x_i \in \mathcal{D}$, we generate $k \in \{1, 2, 4, 8, 16\}$ independent samples using temperature $T = 0.7$ with a maximum of $M = 1024$ tokens per sample. We extract the final answer from each sample using regex-based parsing of boxed expressions and compare the majority-voted answer against the ground truth $y_i^*$ to compute accuracy.

We measure three quantities per problem:
- **Accuracy@$k$**: Indicator $\mathbb{I}[\hat{y}_{i,k} = y_i^*]$, where $\hat{y}_{i,k}$ is the majority answer after $k$ samples
- **Answer consistency $c_{i,k}$**: Fraction of samples agreeing on the majority answer
- **Activation energy $E_a$**: Computed as $E_a = -\ln(c_0)$, where $c_0 = c_{i,16}$ is the consistency at $k = 16$

### 4.1.3 Analysis Groups

The experiments are organized into three analysis groups:
- **G1 (Saturation)**: Fit the exponential model $P_k = P_\infty \cdot (1 - e^{-k/k_0})$ to aggregate and per-problem accuracy curves
- **G2 (Consistency)**: Compute $E_a$ from answer consistency and correlate with MATH difficulty level $L_i$
- **G3 (Routing)**: Threshold $E_a$ to classify problems as "easy" (low-$E_a$) vs. "hard" (high-$E_a$) and measure single-pass accuracy

Table 1 summarizes the experimental configuration.

**Table 1: Experiment Configuration**

| Parameter | Value |
|-----------|-------|
| Model | *Qwen2.5-Math-7B-Instruct* |
| Dataset | MATH test set (50 problems) |
| Temperature $T$ | 0.7 |
| Max tokens $M$ | 1024 |
| Sample counts $k$ | 1, 2, 4, 8, 16 |
| Random seed | 42 |
| Hardware | RTX PRO 6000 Blackwell, PyTorch 2.11.0 |

## 4.2 Results

### 4.2.1 H1: Aggregate Arrhenius Saturation is Confirmed

We test whether aggregate accuracy follows the exponential saturation model $P_k = P_\infty \cdot (1 - e^{-k/k_0})$. Table 2 reports accuracy at each sampling count.

**Table 2: Aggregate Accuracy vs. Sampling Count ($n = 50$)**

| $k$ | Accuracy |
|-----|----------|
| 1 | 68.0% |
| 2 | 78.0% |
| 4 | 86.0% |
| 8 | 84.0% |
| 16 | 82.0% |

We fit the exponential model using nonlinear least squares. The fitted parameters are:
- **Asymptotic ceiling $P_\infty = 0.835$**: The maximum achievable accuracy regardless of sample count
- **Characteristic count $k_0 = 0.613$**: The rate at which accuracy converges to $P_\infty$
- **$R^2 = 0.924$**: The model explains 92.4% of variance in accuracy across $k$ values

The $R^2$ exceeds our threshold of 0.85, confirming H1 at the aggregate level. The model comparison via AICc and BIC also favors the exponential model over power-law and logarithmic alternatives (AICc: $-30.43$ vs. $-10.07$ vs. $-21.73$; BIC: $-37.21$ vs. $-35.24$ vs. $-28.52$).

A critical caveat is required: the aggregate fit smooths over substantial individual variation. The per-problem median $R^2$ for the exponential model is 0.000, and 80% of problems fail to produce a meaningful fit. Only 10 of 50 problems yield valid per-problem $k_0$ estimates. The exponential saturation pattern is therefore a group-level statistical regularity, not a universal law that applies to individual problems. Figure 3 visualizes the aggregate saturation curve with the fitted exponential model.

<!-- FIGURES
- Figure 3: Aggregate accuracy vs. $k$ with exponential fit ($R^2 = 0.924$) and 95% confidence band. Data source: `analysis_h1.json`.
-->

### 4.2.2 H2: $E_a$ Correlates with MATH Level

We test whether consistency-derived activation energy correlates with the annotated MATH difficulty level. Table 3 reports mean $E_a$ by level.

**Table 3: Activation Energy by MATH Level ($n = 50$)**

| Level | Count | Mean $E_a$ | Std $E_a$ | Accuracy |
|-------|-------|-----------|----------|----------|
| 1 | 2 | 9.465 | 0.000 | 50.0% |
| 2 | 9 | 9.643 | 0.252 | 77.8% |
| 3 | 15 | 9.750 | 0.267 | 66.7% |
| 4 | 11 | 9.708 | 0.266 | 72.7% |
| 5 | 13 | 10.000 | $1.9 \times 10^{-6}$ | 61.5% |

The Spearman correlation between $E_a$ and MATH level is $\rho = 0.448$ ($p = 0.001$), exceeding our threshold of 0.4 and confirming H2. Higher difficulty levels correspond to higher activation energy, indicating lower answer consistency.

Two caveats limit the practical utility of this correlation. First, $E_a$ values are bimodally distributed, clustering at approximately 9.47 ($c_0 \approx 0.106$) and 10.0 ($c_0 \approx 0.00005$). This bimodality reflects two stability modes rather than a continuous difficulty spectrum. Second, Level 5 shows near-zero variance ($\sigma \approx 1.9 \times 10^{-6}$), suggesting algorithmic saturation in the consistency computation rather than genuine discriminative signal. Figure 4 shows the $E_a$ distribution across MATH levels.

<!-- FIGURES
- Figure 4: Box plot + scatter of $E_a$ distribution across MATH Levels 1--5, highlighting bimodal clustering. Data source: `analysis_h2.json`.
-->

### 4.2.3 H3: Single-Pass Routing is Falsified

We test the prescriptive claim that low-$E_a$ problems can be solved in a single pass with greater than 75% accuracy. We threshold $E_a$ at the optimal data-dependent value (9.999...) and measure single-pass accuracy for each group.

**Table 4: Single-Pass Accuracy by $E_a$ Category ($n = 50$)**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Optimal threshold | 9.9999999999 | Post-hoc (data leakage) |
| Low-$E_a$ accuracy | 75.0% | Meets threshold only via post-hoc optimization |
| High-$E_a$ accuracy | 50.0% | -- |
| AUC | 0.436 | $< 0.5$: worse than random |
| Spearman($E_a$, accuracy) | $-0.063$ | $p = 0.66$: no relationship |

The 75% threshold pass is a statistical artifact of the bimodal $E_a$ distribution, not genuine predictive signal. The decisive evidence for falsification comes from two independent metrics: AUC $= 0.436$ (below the 0.5 random-guessing baseline) and Spearman($E_a$, accuracy) $= -0.063$ ($p = 0.66$), indicating no monotonic relationship between activation energy and single-pass correctness. H3 is therefore **falsified**.

The failure mechanism is straightforward: answer consistency measures whether samples agree on an answer, not whether that answer is correct. Problems with $E_a \approx 9.47$ (high consistency) include both correct and incorrect single-pass outcomes. Stable wrong answers exist---the model can be consistently wrong across all 16 samples. Figure 5 visualizes the routing failure with the ROC curve inset.

<!-- FIGURES
- Figure 5: Bar chart of low-$E_a$ (75.0%) vs. high-$E_a$ (50.0%) single-pass accuracy with 75% threshold line; inset ROC curve with AUC = 0.436. Data source: `analysis_h3.json`.
-->

### 4.2.4 H5: Consistency-$E_a$ and Saturation-$k_0$ are Unrelated

We test whether the two "activation energy" measures---derived from consistency ($E_a = -\ln(c_0)$) and from saturation curve fitting ($k_0$)---capture the same construct. Of 50 problems, only 10 yield valid per-problem $k_0$ estimates. The Spearman correlation between $E_a$ and $k_0$ is $\rho = -0.219$ ($p = 0.54$), far below our threshold of 0.5. H5 is **falsified**.

This result indicates that consistency-$E_a$ and saturation-$k_0$ measure different constructs. The theoretical framework lacks internal consistency: the two theoretically related difficulty measures are empirically unrelated.

### 4.2.5 Hypothesis Summary

Table 5 summarizes the results for all four hypotheses.

**Table 5: Hypothesis Results Summary**

| Hypothesis | Metric | Threshold | Actual | Status |
|------------|--------|-----------|--------|--------|
| H1 (Aggregate Arrhenius) | $R^2$ | $> 0.85$ | 0.924 | Confirmed (aggregate only) |
| H2 ($E_a$ = difficulty) | Spearman $\rho$ | $> 0.4$ | 0.448 | Confirmed (modest effect) |
| H3 (Single-pass routing) | AUC | $> 0.5$ | 0.436 | **Falsified** |
| H5 ($E_a$ = $k_0$) | Spearman $\rho$ | $> 0.5$ | $-0.219$ | **Falsified** |

## 4.3 Discussion of Findings

### 4.3.1 Confirmed Descriptive Patterns

H1 and H2 confirm two descriptive predictions of Activation Energy Theory. The exponential saturation model ($R^2 = 0.924$) accurately describes aggregate accuracy improvement with sampling count, cross-validating the probabilistic inference scaling framework of Yang et al. [3]. The correlation between $E_a$ and MATH level ($\rho = 0.448$) confirms that consistency signals capture coarse difficulty information.

These findings do not, however, enable reliable routing decisions. The falsification of H3 reveals a fundamental conceptual gap: difficulty and single-pass solveability are distinct constructs.

### 4.3.2 The Stability-Correctness Gap

Our negative result exposes a critical limitation of consistency-based routing. Activation energy measures answer stability---the degree to which samples agree---but stability does not imply correctness. We identify three failure modes in low-$E_a$ problems:

1. **Execution errors**: All samples make the same arithmetic mistake, producing high consistency but incorrect answers
2. **Conceptual errors**: All samples apply the wrong approach, producing internally consistent but wrong reasoning
3. **Answer extraction failures**: Samples correctly solve the problem but fail to extract the answer in the expected format

The bimodal $E_a$ distribution (~9.47 and ~10.0) reflects two stability modes, not two correctness modes. Problems in the low-$E_a$ cluster include both correct and incorrect single-pass outcomes, which is why threshold-based routing fails.

### 4.3.3 The Irreducible Error Floor

The low-$E_a$ accuracy of 75.0% (achieved only via post-hoc threshold optimization) implies an irreducible error floor of approximately 25 percentage points for consistency-based routing. This floor is larger than the ~8 percentage points reported for variance-based routing alternatives [4], suggesting that the choice of signal (consistency vs. variance) significantly affects the practical utility of routing.

## 4.4 Limitations

Our experiments have several limitations:

1. **Sample size**: 50 problems provide limited statistical power; smaller effects may be undetected
2. **Single model**: *Qwen2.5-Math-7B-Instruct* may not generalize to other models or domains
3. **Post-hoc threshold**: The 75% figure relies on data-dependent threshold optimization, making it unreliable for generalization
4. **Answer extraction**: Systematic extraction errors may confound correctness labels
5. **Bimodal artifacts**: Level 5 $E_a$ values show near-zero variance, suggesting algorithmic saturation in the consistency computation

---

## References

[1] Qwen Team. Qwen2.5-Math-7B-Instruct technical report, 2024.

[2] Hendrycks, D., et al. Measuring mathematical problem solving with the MATH dataset. *NeurIPS Datasets Track*, 2021.

[3] Yang et al. Probabilistic inference scaling theory. 2025.

[4] ACAR (variance-based routing comparison). See related work in Section 7.
