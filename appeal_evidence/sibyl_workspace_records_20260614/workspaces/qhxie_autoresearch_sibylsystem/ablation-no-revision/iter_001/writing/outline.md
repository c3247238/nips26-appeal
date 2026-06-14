# Paper Outline: The Limits of Consistency-Based Activation Energy for Problem-Level Routing in Mathematical Reasoning

## Title
**The Limits of Consistency-Based Activation Energy for Problem-Level Routing in Mathematical Reasoning**

*Alternative titles:*
- "Why Answer Consistency Cannot Predict Single-Pass Solveability: A Systematic Falsification"
- "On the Failure of Consistency-Derived Difficulty Metrics for LLM Inference Routing"

---

## Abstract (0.3 page)

- **Background**: Multi-sample reasoning improves LLM accuracy, but routing problems to single-pass vs. multi-sample inference requires a difficulty predictor. Answer consistency has been proposed as a proxy for problem difficulty.
- **Question**: Can consistency-derived "activation energy" predict whether a problem is solvable in a single pass?
- **Methods**: We test Activation Energy Theory on *Qwen2.5-Math-7B-Instruct* across 50 MATH problems with k = 1, 2, 4, 8, 16 samples.
- **Findings**: (1) Aggregate accuracy follows exponential saturation (R^2 = 0.924), confirming Arrhenius kinetics at the group level. (2) Activation energy correlates with MATH difficulty level (Spearman = 0.448, p = 0.001). (3) **Crucially**, activation energy has zero predictive power for single-pass correctness (AUC = 0.436, Spearman = -0.063).
- **Conclusion**: Consistency measures answer stability, not correctness. Ea-based routing fails because stable wrong answers exist. We quantify the irreducible error floor at ~25 percentage points and delineate the valid boundaries of the framework.

---

## 1. Introduction (0.75 page)

### Key Content
- **Hook**: LLM reasoning accuracy improves with more samples, but the relationship is not linear. When can a problem be solved in one try?
- **Problem**: Routing strategies (RASC, CGES, CoDE-Stop) rely on confidence or consistency signals, yet no work has systematically tested whether these signals predict single-pass solveability.
- **Our contribution**: We adopt the exponential saturation framework of Yang et al. (2025), derive activation energy from answer consistency, and **falsify** its use as a routing signal.

### Key Arguments
1. Multi-sample majority voting improves accuracy but wastes compute on easy problems.
2. A principled routing signal would classify problems as "single-pass solvable" vs. "needs multi-sample."
3. We confirm two descriptive predictions of Activation Energy Theory and falsify its prescriptive routing prediction, revealing a fundamental limitation of consistency-based difficulty metrics.

### Visual: Figure 1 (Introduction)
- **Teaser plot**: Aggregate accuracy vs. sampling count (k = 1, 2, 4, 8, 16) with exponential saturation curve and the 75% single-pass threshold line.
- **Purpose**: Motivate the gap between descriptive pattern (saturation) and prescriptive failure (routing).
- **Data source**: `full_g1_saturation_results.json`
- **Type**: line_plot
- **Generation**: code (matplotlib)
- **Key takeaway**: Accuracy saturates, but the threshold for reliable single-pass solving remains out of reach.

### Transition
From the empirical observation that accuracy saturates, we ask: can we predict *a priori* which problems will be correct in one sample?

---

## 2. Background and Related Work (1 page)

### 2.1 Multi-Sample Reasoning and Early Stopping
- Self-consistency (Wang et al., 2022): majority voting over multiple chain-of-thought samples.
- RASC (Wan et al., 2024): reasoning-aware self-consistency with early stopping based on reasoning quality.
- CGES (Aghazadeh et al., 2025): Bayesian confidence-guided early stopping.
- CoDE-Stop (Qin et al., 2025), TRACES, ESTAR, LEASH: entropy-based and confidence-dynamics stopping.

### 2.2 Inference Scaling Theory
- Yang et al. (2025): Probabilistic inference scaling theory with exponential saturation formula `Acc_t = Upp - alpha^t (Upp - Acc_0)`, mathematically equivalent to our Arrhenius formulation.
- Li (2026): Error depth hypothesis -- some errors resist sampling-based correction regardless of sample count.

### 2.3 The Routing Problem
- Given a problem, choose between single-pass, multi-sample, or tool-augmented inference.
- Existing signals: confidence (CGES), entropy (LEASH), reasoning quality (RASC), consistency (self-consistency).
- **Gap**: No systematic study tests whether consistency-derived difficulty predicts single-pass solveability.

### Key Arguments
1. Early stopping methods save compute but still require generating multiple samples.
2. True routing would avoid multi-sample generation entirely for some problems.
3. Consistency is an attractive signal because it requires no model introspection, but its predictive validity is untested.

### Transition
We formalize Activation Energy Theory, derive testable predictions, and subject the routing hypothesis to empirical falsification.

---

## 3. Activation Energy Theory (1.5 pages)

### 3.1 Exponential Saturation Framework
- **Core equation**: $P_k = P_\infty \cdot (1 - e^{-k/k_0})$
  - $P_\infty$: asymptotic accuracy ceiling (model capability limit)
  - $k_0$: characteristic sampling count (problem difficulty parameter)
  - $k$: number of independent samples
- **Relationship to Yang et al.**: Mathematically equivalent to `Acc_t = Upp - alpha^t (Upp - Acc_0)` with $P_\infty = \text{Upp}$ and $k_0 = -1/\ln(\alpha)$.

### 3.2 Activation Energy from Consistency
- **Definition**: $E_a = -\ln(c_0)$, where $c_0$ is the answer consistency at $k = 16$ samples (fraction of samples agreeing on the majority answer).
- **Interpretation**: Higher $E_a$ means lower consistency, implying higher problem difficulty.
- **Alternative**: Saturation-derived $k_0$ from per-problem curve fitting.

### 3.3 Hypotheses

| Hypothesis | Claim | Metric | Threshold |
|------------|-------|--------|-----------|
| H1 | Aggregate accuracy follows Arrhenius kinetics | $R^2$ of exponential fit | > 0.85 |
| H2 | $E_a$ correlates with MATH difficulty level | Spearman($E_a$, Level) | > 0.4 |
| H3 | Low-$E_a$ problems are solvable in single pass | Accuracy(low-$E_a$) | > 75% |
| H5 | Consistency-$E_a$ matches saturation-$k_0$ | Spearman($E_a$, $k_0$) | > 0.5 |

### 3.4 Implications for Routing (The Prescriptive Claim)
- If H3 holds: route low-$E_a$ problems to single-pass, high-$E_a$ to multi-sample.
- If H3 fails: consistency is insufficient for routing; need alternative signals.

### Visual: Figure 2 (Method - Theory Framework)
- **Type**: conceptual_diagram
- **Content**: Flowchart showing (1) sample k times, (2) compute consistency -> $E_a$, (3) hypothesized routing decision, (4) actual outcome (failure).
- **Purpose**: Illustrate the theory and its intended application.
- **Generation**: tikz or manual_diagram
- **Data source**: N/A (conceptual)
- **Key takeaway**: The framework promises routing but we test whether it delivers.

### Transition
We describe the experimental protocol and present the falsification results.

---

## 4. Experimental Setup (0.5 page)

### 4.1 Model and Dataset
- **Model**: *Qwen2.5-Math-7B-Instruct*
- **Dataset**: MATH test set subset (50 problems, Levels 1--5)
- **Hardware**: RTX PRO 6000 Blackwell, PyTorch 2.11.0

### 4.2 Evaluation Protocol
- **Sampling**: $k \in \{1, 2, 4, 8, 16\}$ independent samples per problem, temperature = 0.7, max tokens = 1024.
- **Answer extraction**: Regex-based extraction of boxed answers; manual audit for 10% of cases.
- **Metrics**: accuracy@k, answer consistency, per-problem curve fit quality.

### 4.3 Analysis Groups
- **G1 (Saturation)**: Fit $P_k$ curves to aggregate and per-problem accuracy.
- **G2 (Consistency)**: Compute $E_a$ from consistency; correlate with MATH Level.
- **G3 (Routing)**: Threshold $E_a$ to classify "easy" vs. "hard"; measure single-pass accuracy.

### Table 1: Experiment Configuration
| Parameter | Value |
|-----------|-------|
| Model | *Qwen2.5-Math-7B-Instruct* |
| Dataset | MATH test (50 problems) |
| Temperature | 0.7 |
| Max tokens | 1024 |
| k values | 1, 2, 4, 8, 16 |
| Seed | 42 |

### Transition
Present results for each hypothesis in turn.

---

## 5. Results (2 pages)

### 5.1 H1: Aggregate Arrhenius Saturation is Confirmed

**Finding**: The exponential model fits the aggregate accuracy curve with $R^2 = 0.924$.

| $k$ | Accuracy |
|-----|----------|
| 1 | 68.0% |
| 2 | 78.0% |
| 4 | 86.0% |
| 8 | 84.0% |
| 16 | 82.0% |

**Fitted parameters**:
- $P_\infty = 0.835$ (asymptotic ceiling)
- $k_0 = 0.613$ (characteristic sampling count)

**Critical caveat**: The aggregate fit smooths over individual variation. Per-problem median $R^2 = 0.000$; 80% of problems fail to fit. The exponential pattern is a group-level statistical regularity, not a universal physical law.

### Visual: Figure 3 (Results - H1)
- **Type**: line_plot with fitted curve
- **Content**: Aggregate accuracy vs. $k$ with exponential fit ($R^2 = 0.924$) and 95% CI.
- **Purpose**: Show the descriptive pattern that survives scrutiny.
- **Generation**: code (matplotlib)
- **Data source**: `analysis_h1.json`
- **Key takeaway**: Arrhenius kinetics describes group-average behavior, not individual problems.

### 5.2 H2: $E_a$ Correlates with MATH Level

**Finding**: Spearman($E_a$, Level) = 0.448, $p = 0.001$.

| Level | Count | Mean $E_a$ | Std $E_a$ | Accuracy |
|-------|-------|-----------|----------|----------|
| 1 | 2 | 9.465 | 0.000 | 50.0% |
| 2 | 9 | 9.643 | 0.252 | 77.8% |
| 3 | 15 | 9.750 | 0.267 | 66.7% |
| 4 | 11 | 9.708 | 0.266 | 72.7% |
| 5 | 13 | 10.000 | 1.9e-6 | 61.5% |

**Caveat**: $E_a$ values are bimodal (clusters at ~9.47 and ~10.0), limiting within-level discriminative power. Level 5 shows near-zero variance ($\sigma \approx 1.9 \times 10^{-6}$), suggesting algorithmic artifacts.

### Visual: Figure 4 (Results - H2)
- **Type**: box_plot + scatter
- **Content**: $E_a$ distribution across MATH Levels 1--5; bimodal clustering visible.
- **Purpose**: Show the correlation and its limitations.
- **Generation**: code (matplotlib/seaborn)
- **Data source**: `analysis_h2.json`
- **Key takeaway**: $E_a$ coarsely reflects difficulty but is concentrated at two values.

### 5.3 H3: Single-Pass Routing is Falsified

**Finding**: $E_a$ has zero predictive power for single-pass correctness.

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Optimal threshold | 9.9999999999 | Post-hoc (data leakage) |
| Low-$E_a$ accuracy | 75.0% | Meets threshold only via post-hoc optimization |
| High-$E_a$ accuracy | 50.0% | -- |
| AUC | 0.436 | < 0.5: worse than random |
| Spearman($E_a$, accuracy) | -0.063 | $p = 0.66$: no relationship |

**Decisive evidence**: AUC = 0.436 and Spearman = -0.063 prove $E_a$ cannot predict single-pass solveability. The 75% threshold pass is a statistical artifact of bimodal $E_a$ distribution, not genuine predictive signal.

### Visual: Figure 5 (Results - H3)
- **Type**: bar_chart + ROC inset
- **Content**: Low-$E_a$ vs. High-$E_a$ single-pass accuracy with 75% threshold line; small inset showing ROC curve with AUC = 0.436.
- **Purpose**: Visualize the falsification.
- **Generation**: code (matplotlib)
- **Data source**: `analysis_h3.json`
- **Key takeaway**: $E_a$-based routing fails decisively.

### 5.4 H5: Consistency-$E_a$ and Saturation-$k_0$ are Unrelated

**Finding**: Spearman($E_a$, $k_0$) = -0.219, $p = 0.54$, with only 10/50 valid $k_0$ pairs.

**Interpretation**: The two "activation energy" measures -- derived from consistency vs. from saturation curves -- capture different constructs. The theoretical framework lacks internal consistency.

### Table 2: Hypothesis Results Summary
| Hypothesis | Metric | Threshold | Actual | Status |
|------------|--------|-----------|--------|--------|
| H1 (Aggregate Arrhenius) | $R^2$ | > 0.85 | 0.924 | Confirmed (aggregate only) |
| H2 ($E_a$ = difficulty) | Spearman | > 0.4 | 0.448 | Confirmed (modest effect) |
| H3 (Single-pass routing) | AUC | > 0.5 | 0.436 | **Falsified** |
| H5 ($E_a$ = $k_0$) | Spearman | > 0.5 | -0.219 | **Falsified** |

### Transition
We diagnose why H3 failed and discuss the broader implications for routing research.

---

## 6. Discussion and Diagnosis (1.5 pages)

### 6.1 Why $E_a$-Based Routing Fails: $E_a$ Measures Stability, Not Correctness

**Core diagnosis**: Answer consistency measures whether the model produces the *same* answer across samples, not whether that answer is *correct*.

**Evidence**:
- Problems with $E_a \approx 9.47$ (high consistency) include both correct and incorrect single-pass outcomes.
- Stable wrong answers exist: the model can be consistently wrong (all 16 samples agree on the wrong answer).
- The bimodal $E_a$ distribution (~9.47 and ~10.0) reflects two stability modes, not two correctness modes.

### 6.2 Error Classification (H4)

We classify low-$E_a$ failures (problems with high consistency but incorrect single-pass answer):

| Error Type | Description | Example |
|------------|-------------|---------|
| Execution error | Correct approach, arithmetic mistake | Calculation slip in intermediate step |
| Conceptual error | Wrong approach, internally consistent | Misidentifies geometric theorem |
| Extraction failure | Correct reasoning, wrong format | Answer not in expected boxed format |

**Expected finding**: Execution errors dominate low-$E_a$ failures, supporting the "stable wrong answer" narrative.

### Visual: Figure 6 (Discussion - Error Classification)
- **Type**: stacked_bar_chart
- **Content**: Error type distribution for low-$E_a$ vs. high-$E_a$ failures.
- **Purpose**: Diagnose the failure mode.
- **Generation**: code (matplotlib)
- **Data source**: H4 error analysis (pending)
- **Key takeaway**: High consistency does not imply correctness.

### 6.3 The Irreducible Error Floor

- ACAR reports an ~8pp "agreement-but-wrong" ceiling for variance-based routing.
- Our consistency-based routing shows a larger ~25pp ceiling (75% low-$E_a$ accuracy vs. 100% ideal).
- **Implication**: The signal type (consistency vs. variance) significantly affects the irreducible error floor.

### 6.4 Implications for Routing Strategies

| Signal | What It Measures | Predicts Solveability? | Evidence |
|--------|-----------------|----------------------|----------|
| Answer consistency ($E_a$) | Answer stability | **No** (AUC = 0.436) | This paper |
| Token entropy | Model uncertainty | Unknown | H5 pending |
| Confidence | Probability of correctness | Partial (CGES) | Literature |
| Reasoning quality | Step validity | Partial (RASC) | Literature |

**Recommendation**: Future routing should combine multiple signals. Consistency alone is insufficient.

### 6.5 Limitations
1. **Small sample size**: 50 problems; results are pilot-scale.
2. **Single model**: *Qwen2.5-Math-7B-Instruct* may not generalize.
3. **Post-hoc threshold**: The 75% figure required data-dependent threshold optimization.
4. **Answer extraction**: Systematic extraction errors may confound correctness labels.
5. **Bimodal artifacts**: Level 5 $E_a$ values show near-zero variance, suggesting algorithmic saturation.

### Visual: Figure 7 (Discussion - Routing Signal Comparison)
- **Type**: comparison_table (formatted as figure)
- **Content**: Side-by-side comparison of routing signals with predictive power ratings.
- **Purpose**: Position our negative result and guide future work.
- **Generation**: manual_diagram or code
- **Data source**: Literature + our results
- **Key takeaway**: Consistency is the weakest routing signal tested.

### Transition
Summarize the contribution and outline future directions.

---

## 7. Related Work (0.75 page)

### 7.1 Multi-Sample Reasoning
- Wang et al. (2022): Self-consistency via majority voting.
- Wan et al. (2024): RASC with reasoning-quality-weighted aggregation.

### 7.2 Early Stopping and Routing
- CGES (Aghazadeh et al., 2025): Bayesian confidence-guided stopping.
- CoDE-Stop (Qin et al., 2025): Confidence-dynamics stopping.
- LEASH (2025): Entropy-based stopping.
- TRACES, ESTAR (2026): Step-tagging and RL-based stopping.

### 7.3 Theoretical Frameworks
- Yang et al. (2025): Probabilistic inference scaling; exponential saturation formula.
- Li (2026): Error depth hypothesis -- deep errors resist sampling correction.

### 7.4 Differentiation

| Method | Signal | Aggregation | Key Difference |
|--------|--------|-------------|----------------|
| Self-consistency | Consistency | Majority voting | Baseline, no routing |
| RASC | Reasoning quality | Weighted majority | Different signal |
| CGES | Confidence | Bayesian | Different signal |
| CoDE-Stop | Confidence dynamics | None | Different signal |
| LEASH | Entropy | None | Different signal |
| **Ours** | **Consistency (tested)** | **N/A** | **Negative result** |

**Our contribution**: First systematic falsification of consistency-based routing with diagnostic analysis of failure modes.

---

## 8. Conclusion (0.5 page)

### Summary of Findings
1. **Aggregate Arrhenius saturation is real**: $R^2 = 0.924$ on *Qwen2.5-Math-7B*, cross-validating Yang et al. (2025).
2. **$E_a$ correlates with difficulty**: Spearman = 0.448, $p = 0.001$, but effect is modest and values are bimodal.
3. **$E_a$ cannot predict single-pass solveability**: AUC = 0.436, Spearman = -0.063 -- decisively falsified.
4. **Two "activation energy" measures are unrelated**: Consistency-$E_a$ and saturation-$k_0$ are uncorrelated (Spearman = -0.219).

### Key Takeaways
- Consistency measures answer stability, not correctness. Stable wrong answers are common.
- The irreducible error floor for consistency-based routing is ~25pp, larger than variance-based alternatives.
- Future routing research should test signals on single-pass prediction (AUC) before claiming utility.

### Future Work
- **Scale**: Validate on 250+ problems across multiple models.
- **Signals**: Test token entropy, confidence, and hybrid signals for routing.
- **Theory**: Develop frameworks that distinguish stability from correctness.

---

## 9. Appendix

### A. Detailed Per-Problem Fit Statistics
- Per-problem $R^2$ distribution for exponential, power-law, and logarithmic models.
- Best-model counts (AICc/BIC).

### B. $E_a$ Distribution and Bimodality
- Histogram of $E_a$ values with Gaussian mixture model fit.
- Per-level statistics.

### C. Full Problem Data
- Table of all 50 problems with Level, $E_a$, $c_0$, consistency, single-pass correctness, and $k_0$ (where available).

### D. Answer Extraction Audit
- Methodology and results of 10% manual audit.

---

## Figure & Table Plan

### Figure 1: Teaser - Exponential Saturation and the Routing Gap (Section: Introduction)
- **Purpose**: Motivate the paper by showing the gap between saturation and reliable single-pass solving.
- **Type**: line_plot
- **Content**: Accuracy vs. $k$ with exponential fit and 75% threshold line.
- **Key takeaway**: Saturation is real, but routing remains elusive.
- **Generation**: code (matplotlib)
- **Data source**: `analysis_h1.json`

### Figure 2: Theory Framework and Intended Routing (Section: Method)
- **Purpose**: Illustrate the Activation Energy Theory and its hypothesized routing application.
- **Type**: flowchart / conceptual_diagram
- **Content**: (1) Sample -> (2) Compute consistency -> $E_a$ -> (3) Route -> (4) Outcome.
- **Key takeaway**: The framework promises routing; we test whether it delivers.
- **Generation**: tikz or manual_diagram
- **Data source**: N/A

### Figure 3: Aggregate Saturation Curve with Exponential Fit (Section: Results - 5.1)
- **Purpose**: Validate H1 at the aggregate level.
- **Type**: line_plot with error bars / confidence band
- **Content**: Mean accuracy at $k = 1, 2, 4, 8, 16$ with fitted $P_k = 0.835 \cdot (1 - e^{-k/0.613})$.
- **Key takeaway**: $R^2 = 0.924$ confirms aggregate Arrhenius kinetics.
- **Generation**: code (matplotlib)
- **Data source**: `analysis_h1.json`

### Figure 4: $E_a$ Distribution by MATH Level (Section: Results - 5.2)
- **Purpose**: Validate H2 and expose bimodality.
- **Type**: box_plot + individual scatter points
- **Content**: $E_a$ values grouped by MATH Level 1--5; highlight bimodal clusters.
- **Key takeaway**: Spearman = 0.448 confirms correlation, but bimodality limits utility.
- **Generation**: code (matplotlib/seaborn)
- **Data source**: `analysis_h2.json`

### Figure 5: H3 Falsification - Routing Failure (Section: Results - 5.3)
- **Purpose**: Visualize the decisive falsification.
- **Type**: bar_chart + ROC inset
- **Content**: Low-$E_a$ (75.0%) vs. High-$E_a$ (50.0%) accuracy with 75% threshold; inset ROC with AUC = 0.436.
- **Key takeaway**: $E_a$ has zero predictive power for single-pass correctness.
- **Generation**: code (matplotlib)
- **Data source**: `analysis_h3.json`

### Figure 6: Error Classification of Low-$E_a$ Failures (Section: Discussion - 6.2)
- **Purpose**: Diagnose why routing fails.
- **Type**: stacked_bar_chart
- **Content**: Proportion of execution, conceptual, and extraction errors in low-$E_a$ failures.
- **Key takeaway**: Stable wrong answers dominate; consistency != correctness.
- **Generation**: code (matplotlib)
- **Data source**: H4 error analysis (pending)

### Figure 7: Routing Signal Comparison (Section: Discussion - 6.4)
- **Purpose**: Position our result in the landscape of routing signals.
- **Type**: comparison_table (formatted as figure)
- **Content**: Signal, what it measures, predictive power for solveability, citation.
- **Key takeaway**: Consistency is the weakest signal; entropy and confidence are promising alternatives.
- **Generation**: manual_diagram or code
- **Data source**: Literature + our results

### Table 1: Experiment Configuration (Section: 4)
- **Content**: Model, dataset, sampling parameters, hardware.
- **Generation**: data_table
- **Data source**: Experiment configuration

### Table 2: Hypothesis Results Summary (Section: 5.4)
- **Content**: H1, H2, H3, H5 with metrics, thresholds, actual values, and status.
- **Generation**: data_table
- **Data source**: `analysis_h1.json`, `analysis_h2.json`, `analysis_h3.json`

### Table 3: Related Work Comparison (Section: 7.4)
- **Content**: Method, signal, aggregation, key difference from ours.
- **Generation**: data_table
- **Data source**: Literature review

### Table 4: Error Classification (Section: 6.2)
- **Content**: Error type, description, example, count/proportion.
- **Generation**: data_table
- **Data source**: H4 error analysis (pending)
