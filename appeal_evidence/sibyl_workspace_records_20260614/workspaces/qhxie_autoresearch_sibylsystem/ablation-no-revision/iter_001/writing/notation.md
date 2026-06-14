# Notation Table: The Limits of Consistency-Based Activation Energy for Problem-Level Routing

## Model and Problem Parameters

| Symbol | Definition | Dimensions | Notes |
|--------|------------|------------|-------|
| $\mathcal{M}$ | Language model | -- | *Qwen2.5-Math-7B-Instruct* |
| $\mathcal{D}$ | Problem dataset | $n$ problems | MATH test set subset |
| $x_i$ | Input problem $i$ | string | $i = 1, \ldots, n$; $n = 50$ |
| $y_i^*$ | Ground-truth answer for $x_i$ | string | Extracted from boxed expression |
| $L_i$ | MATH difficulty level | $\{1, 2, 3, 4, 5\}$ | 1 = pre-algebra, 5 = precalculus |

## Sampling Parameters

| Symbol | Definition | Dimensions | Notes |
|--------|------------|------------|-------|
| $k$ | Number of samples per problem | scalar | $k \in \{1, 2, 4, 8, 16\}$ |
| $S_{i,k}$ | Set of $k$ samples for problem $i$ | $k$ strings | Independent temperature sampling |
| $a_{i,j}$ | Answer from sample $j$ for problem $i$ | string | $j = 1, \ldots, k$ |
| $T$ | Sampling temperature | scalar | $T = 0.7$ |
| $M$ | Maximum tokens per sample | scalar | $M = 1024$ |

## Saturation and Accuracy

| Symbol | Definition | Dimensions | Notes |
|--------|------------|------------|-------|
| $P_k$ | Probability of correct answer after $k$ samples | scalar $\in [0, 1]$ | $P_k = P_\infty \cdot (1 - e^{-k/k_0})$ |
| $P_\infty$ | Asymptotic accuracy ceiling | scalar $\in [0, 1]$ | Model capability limit; estimated 0.835 |
| $k_0$ | Characteristic sampling count | scalar | Problem difficulty from saturation curve |
| $\hat{P}_\infty$ | Estimated asymptotic ceiling | scalar | From non-linear least squares fit |
| $\hat{k}_0$ | Estimated characteristic count | scalar | From non-linear least squares fit |
| $\text{Acc}(S_{i,k})$ | Accuracy indicator for problem $i$ at $k$ | $\{0, 1\}$ | 1 if majority answer matches $y_i^*$, else 0 |

## Consistency and Activation Energy

| Symbol | Definition | Dimensions | Notes |
|--------|------------|------------|-------|
| $c_{i,k}$ | Answer consistency at $k$ samples | scalar $\in [0, 1]$ | Fraction of samples agreeing on majority answer |
| $c_0$ | Final consistency at $k = 16$ | scalar $\in [0, 1]$ | Used for $E_a$ estimation |
| $E_a$ | Activation energy (consistency-derived) | scalar | $E_a = -\ln(c_0)$; higher = lower consistency |
| $\hat{E}_a$ | Estimated activation energy | scalar | From consistency signals |
| $E_a^{\text{low}}$ | Low-$E_a$ threshold | scalar | Below this: hypothesized single-pass routing |
| $E_a^{\text{high}}$ | High-$E_a$ threshold | scalar | Above this: hypothesized multi-sample routing |

## Routing and Prediction

| Symbol | Definition | Dimensions | Notes |
|--------|------------|------------|-------|
| $\tau$ | Single-pass accuracy threshold | scalar | $\tau = 0.75$ (75%) |
| $R(x)$ | Routing decision function | $\{\text{single-pass}, \text{multi-sample}\}$ | Based on $E_a$ threshold |
| $\hat{y}_{i,1}$ | Single-pass answer for problem $i$ | string | Answer from $k = 1$ sample |
| $\mathbb{I}[\cdot]$ | Indicator function | $\{0, 1\}$ | 1 if condition true, else 0 |

## Statistical Metrics

| Symbol | Definition | Dimensions | Notes |
|--------|------------|------------|-------|
| $R^2$ | Coefficient of determination | scalar $\in [0, 1]$ | Goodness of fit; aggregate = 0.924 |
| $\rho$ | Spearman rank correlation | scalar $\in [-1, 1]$ | Monotonic relationship measure |
| $r$ | Pearson correlation | scalar $\in [-1, 1]$ | Linear relationship measure |
| $p$ | p-value | scalar $\in [0, 1]$ | Statistical significance |
| $\text{AUC}$ | Area under ROC curve | scalar $\in [0, 1]$ | Classification performance; 0.5 = random |
| $\text{AIC}$ | Akaike information criterion | scalar | Model comparison (lower = better) |
| $\text{AICc}$ | Corrected AIC | scalar | Adjusted for small sample size |
| $\text{BIC}$ | Bayesian information criterion | scalar | Model comparison with penalty |
| $n$ | Sample size | integer | Number of problems |
| $\mu$ | Mean | scalar | Average value |
| $\sigma$ | Standard deviation | scalar | Spread of values |

## Model Comparison

| Symbol | Definition | Notes |
|--------|------------|-------|
| $f_{\text{exp}}$ | Exponential model | $P_k = P_\infty \cdot (1 - e^{-k/k_0})$ |
| $f_{\text{pow}}$ | Power-law model | $P_k = a \cdot k^b + c$ |
| $f_{\text{log}}$ | Logarithmic model | $P_k = a \cdot \ln(k) + b$ |

## Acronyms

| Acronym | Expansion | Context |
|---------|-----------|---------|
| AUC | Area Under the Curve | Classification performance |
| CoT | Chain-of-Thought | Reasoning prompting |
| CGES | Confidence-Guided Early Stopping | Related work method |
| Ea | Activation Energy | Problem difficulty measure |
| LLM | Large Language Model | General term |
| MATH | Mathematics dataset | Benchmark dataset |
| RASC | Reasoning-Aware Self-Consistency | Related work method |
| ROC | Receiver Operating Characteristic | Classification curve |
| SC | Self-Consistency | Majority voting baseline |

---

## Usage Notes

1. **$E_a$ vs. $k_0$**: These are two distinct measures of "difficulty." $E_a$ is derived from answer consistency ($-\ln(c_0)$); $k_0$ is derived from saturation curve fitting. They are uncorrelated (Spearman = -0.219), meaning they capture different constructs.

2. **Aggregate vs. per-problem**: $R^2 = 0.924$ refers to the aggregate fit over 50 problems. Per-problem median $R^2 = 0.000$, with 80% of problems failing to fit. Always distinguish aggregate from individual when reporting.

3. **Bimodal $E_a$**: $E_a$ clusters at ~9.47 ($c_0 \approx 0.106$) and ~10.0 ($c_0 \approx 0.00005$). This bimodality reflects two stability modes, not two difficulty modes.

4. **Consistency != correctness**: $c_{i,k} = 1.0$ means all samples agree; it does not mean the agreed answer is correct. This is the central theoretical insight of the paper.

5. **Threshold notation**: The optimal $E_a$ threshold (9.999...) is a post-hoc data-dependent value. It should not be reported as a generalizable constant.
