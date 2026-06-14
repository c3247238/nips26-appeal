# 3 Activation Energy Theory

## 3.1 Exponential Saturation Framework

We model the probability that a language model produces a correct answer as a function of the number of independent samples $k$. Following Yang et al. (2025), we adopt an exponential saturation form mathematically equivalent to Arrhenius-like kinetics:

$$P_k = P_\infty \cdot \left(1 - e^{-k / k_0}\right)$$

where $P_\infty \in [0, 1]$ is the asymptotic accuracy ceiling (the model's capability limit for the problem distribution) and $k_0 > 0$ is the characteristic sampling count controlling the rate of convergence. The product $P_\infty \cdot k_0$ has units of samples and can be interpreted as a problem-difficulty parameter: larger values indicate slower convergence toward the ceiling.

This formulation is mathematically equivalent to the probabilistic inference scaling theory of Yang et al. (2025), who write $\text{Acc}_t = \text{Upp} - \alpha^t (\text{Upp} - \text{Acc}_0)$. The correspondence is $P_\infty = \text{Upp}$ and $k_0 = -1 / \ln(\alpha)$. We use the Arrhenius form because it separates the ceiling ($P_\infty$) from the rate ($k_0$) and admits a direct physical analogy.

## 3.2 Activation Energy from Consistency

We derive a problem-level difficulty measure from answer consistency rather than from per-problem curve fitting. For problem $x_i$, let $S_{i,k} = \{a_{i,1}, \ldots, a_{i,k}\}$ denote the set of $k$ sampled answers. The answer consistency $c_{i,k}$ is the fraction of samples that agree with the majority answer:

$$c_{i,k} = \frac{1}{k} \max_{a} \sum_{j=1}^{k} \mathbb{I}[a_{i,j} = a]$$

We define the consistency-derived activation energy as

$$E_a = -\ln(c_0)$$

where $c_0 = c_{i,16}$ is the answer consistency at $k = 16$ samples. Higher $E_a$ indicates lower consistency and, by hypothesis, higher problem difficulty. We also compute a saturation-derived $k_0$ for each problem by fitting the exponential model to its per-problem accuracy trajectory; this yields an alternative difficulty measure for comparison (H5).

**Critical distinction.** Answer consistency measures whether the model produces the *same* answer across samples, not whether that answer is *correct*. A problem can have $c_0 = 1.0$ (all samples agree) yet be consistently wrong. This distinction is central to the falsification in Section 5.

## 3.3 Hypotheses

We derive four testable hypotheses from the framework:

| Hypothesis | Claim | Metric | Threshold |
|------------|-------|--------|-----------|
| H1 | Aggregate accuracy follows Arrhenius kinetics | $R^2$ of exponential fit | $> 0.85$ |
| H2 | $E_a$ correlates with MATH difficulty level | Spearman($E_a$, Level) | $> 0.4$ |
| H3 | Low-$E_a$ problems are solvable in a single pass | Accuracy(low-$E_a$) | $> 75\%$ |
| H5 | Consistency-$E_a$ matches saturation-$k_0$ | Spearman($E_a$, $k_0$) | $> 0.5$ |

H1 and H2 are descriptive predictions: they test whether the exponential framework and the consistency-derived difficulty measure capture observable structure in the data. H3 is the prescriptive routing prediction: if low-$E_a$ problems achieve high single-pass accuracy, then $E_a$ can be used to route problems to single-pass vs. multi-sample inference. H5 tests the internal consistency of the theory by comparing two independent difficulty estimates.

## 3.4 Implications for Routing

If H3 holds, a natural routing strategy emerges: classify problems with $E_a$ below a threshold $E_a^{\text{low}}$ as "single-pass solvable" and problems with $E_a$ above $E_a^{\text{high}}$ as "multi-sample required." The threshold can be set by requiring a minimum single-pass accuracy $\tau = 0.75$ (75%).

If H3 fails, the implication is that answer consistency is insufficient as a routing signal. The model may produce stable wrong answers---high consistency, low accuracy---rendering $E_a$ useless for predicting single-pass solveability. We test this decisively in Section 5.

Figure 2 illustrates the hypothesized pipeline: sample $k$ times, compute consistency, derive $E_a$, threshold for routing, and evaluate the outcome.

---

# 4 Experimental Setup

## 4.1 Model and Dataset

We evaluate on *Qwen2.5-Math-7B-Instruct* [Qwen Team, 2024], a 7-billion parameter instruction-tuned model specialized for mathematics. We use the MATH test set [Hendrycks et al., 2021], a benchmark of 12,500 competition mathematics problems with five difficulty levels (Level 1: pre-algebra through Level 5: precalculus). We select a stratified subset of 50 problems spanning Levels 1--5.

All experiments are conducted on a single NVIDIA RTX PRO 6000 Blackwell GPU (compute capability sm_120) using PyTorch 2.11.0, which introduced native Blackwell support.

## 4.2 Evaluation Protocol

For each problem $x_i$, we generate $k \in \{1, 2, 4, 8, 16\}$ independent samples using temperature $T = 0.7$ and a maximum of $M = 1024$ tokens per sample. The random seed is fixed at 42 for reproducibility.

**Answer extraction.** We extract final answers from the model output using a regex-based parser that matches boxed expressions (e.g., `\boxed{...}`). We manually audit 10% of extractions to verify parser accuracy.

**Accuracy computation.** At each $k$, the majority-voted answer across the $k$ samples is compared against the ground-truth answer $y_i^*$ to compute the accuracy indicator $\text{Acc}(S_{i,k}) \in \{0, 1\}$. Aggregate accuracy at each $k$ is the mean over all 50 problems.

**Consistency computation.** For each problem and each $k$, we compute $c_{i,k}$ as the fraction of samples agreeing on the majority answer. The final consistency $c_0 = c_{i,16}$ is used to compute $E_a = -\ln(c_0)$.

## 4.3 Analysis Groups

We define three analysis groups aligned with our hypotheses:

**G1 (Saturation).** We fit the exponential model $P_k = P_\infty \cdot (1 - e^{-k/k_0})$ to the aggregate accuracy trajectory across all 50 problems using non-linear least squares. H1 is confirmed if the fitted $R^2 > 0.85$. We additionally fit the model per-problem and report the distribution of $R^2$ values to test whether the aggregate pattern holds at the individual level.

**G2 (Consistency).** For each problem, we compute $c_0$ at $k = 16$ and derive $E_a = -\ln(c_0)$. We test H2 by computing Spearman $\rho(E_a, L_i)$ against the MATH difficulty level $L_i \in \{1, 2, 3, 4, 5\}$. H2 is confirmed if $\rho > 0.4$. We also test H5 by computing Spearman $\rho(E_a, k_0)$ for problems with valid per-problem saturation fits.

**G3 (Routing).** We use the $E_a$ estimates from G2 to classify problems into low-$E_a$ and high-$E_a$ subsets. The threshold is optimized post-hoc on the full dataset to maximize single-pass accuracy of the low-$E_a$ group. We report the resulting single-pass accuracy, AUC of the $E_a$-based classifier, and Spearman $\rho(E_a, \text{single-pass correctness})$. H3 is confirmed only if the low-$E_a$ accuracy exceeds 75% with a genuinely predictive threshold (AUC $> 0.5$).

Table 1 summarizes the experiment configuration.

**Table 1: Experiment configuration.**

| Parameter | Value |
|-----------|-------|
| Model | *Qwen2.5-Math-7B-Instruct* |
| Dataset | MATH test (50 problems, Levels 1--5) |
| Temperature | 0.7 |
| Max tokens | 1024 |
| $k$ values | 1, 2, 4, 8, 16 |
| Seed | 42 |
| Hardware | RTX PRO 6000 Blackwell (sm_120) |
| PyTorch version | 2.11.0 |
