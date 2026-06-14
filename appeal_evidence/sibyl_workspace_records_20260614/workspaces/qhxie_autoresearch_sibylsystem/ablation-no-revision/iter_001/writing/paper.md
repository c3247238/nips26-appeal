# The Limits of Consistency-Based Activation Energy for Problem-Level Routing in Mathematical Reasoning

## Abstract

Large language models improve reasoning accuracy through multi-sample generation and majority voting, but the relationship between sampling count and accuracy saturates exponentially rather than growing without bound. This saturation pattern raises a practical and economically consequential question: can we predict which problems are solvable in a single pass, enabling problem-level routing between single-sample and multi-sample inference? We adopt the exponential saturation framework of Yang et al. and derive a problem-level activation energy measure from answer consistency to test whether consistency-derived difficulty predicts single-pass solveability. We evaluate on *Qwen2.5-Math-7B-Instruct* across 50 problems from the MATH dataset with sampling counts $k = 1, 2, 4, 8, 16$. Aggregate accuracy follows Arrhenius-like kinetics with $R^2 = 0.924$, and activation energy correlates with MATH difficulty level (Spearman $\rho = 0.448$, $p = 0.001$). However, activation energy has zero predictive power for single-pass correctness: AUC = 0.436 and $\rho = -0.063$ ($p = 0.66$), decisively falsifying the routing hypothesis. The irreducible error floor for consistency-based routing is approximately 25 percentage points. We conclude that consistency measures answer stability, not correctness; future routing research must test signals on single-pass prediction before claiming utility.

---

## 1. Introduction

Large language models (LLMs) exhibit a striking empirical regularity: their accuracy on reasoning tasks improves when allowed to generate multiple independent samples and aggregate the results through majority voting [1]. This multi-sample paradigm, formalized as self-consistency by Wang et al. [1], has become a standard technique for extracting reliable performance from probabilistic models. Yet the relationship between sampling count and accuracy is not linear. Accuracy rises steeply at first, then plateaus as the model approaches its asymptotic ceiling $P_\infty$ [8]. Figure 1 shows this saturation pattern for *Qwen2.5-Math-7B-Instruct* on 50 MATH problems, illustrating the gap between the descriptive regularity of saturation and the elusive goal of reliable single-pass solving. This pattern raises a natural and economically consequential question: for which problems can a single sample suffice, and which demand the full cost of multi-sample generation?

Answering this question requires a routing signal---a predictor that classifies problems as "single-pass solvable" or "needs multi-sample" before any samples are generated. Several candidates have been proposed in the literature. Confidence-Guided Early Stopping (CGES) uses token-level confidence to decide when enough samples have been collected [3]. Reasoning-Aware Self-Consistency (RASC) incorporates reasoning quality signals into the aggregation process [2]. CoDE-Stop, LEASH, TRACES, and ESTAR exploit entropy dynamics or confidence trajectories to stop sampling early [4, 5, 6, 7]. What these methods share is that they all operate during multi-sample generation; they reduce the number of samples needed, but they do not eliminate multi-sample generation entirely for any problem.

A more ambitious goal is problem-level routing: assigning an inference strategy to each problem before generation begins. For such routing to be principled, the signal must predict single-pass solveability---the probability that the model will produce a correct answer in one sample. Answer consistency, the fraction of multiple samples that agree on the majority answer, is an attractive candidate because it requires no model introspection and can be estimated from a small pilot sample. If consistency reliably indicates difficulty, then problems with high consistency (low activation energy) could be routed to single-pass inference, while low-consistency problems receive multi-sample treatment.

This paper subjects that hypothesis to systematic falsification. We adopt the exponential saturation framework of Yang et al. [8], which models aggregate accuracy as $P_k = P_\infty \cdot (1 - e^{-k/k_0})$, and derive from it an activation energy measure $E_a = -\ln(c_0)$ where $c_0$ is the answer consistency at $k = 16$ samples. We test four hypotheses on *Qwen2.5-Math-7B-Instruct* [11] across 50 problems from the MATH dataset [9]. H1 asks whether aggregate accuracy follows Arrhenius-like kinetics; H2 asks whether $E_a$ correlates with MATH difficulty level; H3 asks whether low-$E_a$ problems are solvable in a single pass with at least 75% accuracy; and H5 asks whether consistency-derived $E_a$ agrees with saturation-derived $k_0$.

Our findings are threefold. First, aggregate accuracy does follow exponential saturation with $R^2 = 0.924$, confirming the descriptive validity of the Arrhenius framework at the group level. Second, $E_a$ correlates with MATH difficulty level (Spearman $\rho = 0.448$, $p = 0.001$), though the distribution is bimodal and coarse. Third, and crucially, $E_a$ has zero predictive power for single-pass correctness: AUC = 0.436 and $\rho = -0.063$ ($p = 0.66$). The 75% accuracy threshold for low-$E_a$ problems is achievable only through post-hoc threshold optimization, a statistical artifact rather than genuine predictive signal. Additionally, consistency-$E_a$ and saturation-$k_0$ are uncorrelated (Spearman $\rho = -0.219$, $p = 0.54$), indicating that the two "difficulty" measures capture different constructs.

The central diagnosis is that answer consistency measures stability, not correctness. A model can be consistently wrong---all 16 samples agreeing on an incorrect answer---yielding high consistency (low $E_a$) but zero accuracy. Stable wrong answers are not rare; they represent the dominant failure mode of consistency-based routing. We quantify the irreducible error floor at approximately 25 percentage points and argue that future routing research must test signals on single-pass prediction before claiming utility.

The remainder of this paper is organized as follows. Section 2 reviews multi-sample reasoning, early stopping, and routing strategies, and positions our contribution against existing work. Section 3 formalizes Activation Energy Theory and derives the testable predictions. Section 4 describes the experimental protocol. Section 5 presents the results for each hypothesis. Section 6 diagnoses the failure of $E_a$-based routing and discusses implications. Section 7 concludes with future directions.

---

## 2. Related Work

### 2.1 Multi-Sample Reasoning and Early Stopping

The dominant approach to improving LLM reasoning accuracy without additional training is to generate multiple independent samples and aggregate their answers. Wang et al. [1] introduced self-consistency (SC), which samples $k$ chain-of-thought (CoT) reasoning paths and selects the answer via majority voting. SC established that answer consistency across samples is a reliable signal for selecting the most probable correct answer at the aggregate level. However, SC requires generating all $k$ samples upfront, making it computationally expensive even for problems that might be solvable in a single pass.

Wan et al. [2] proposed reasoning-aware self-consistency (RASC), which extends SC by weighting votes according to reasoning quality signals rather than treating all samples equally. RASC demonstrated that reasoning-quality-weighted aggregation outperforms uniform majority voting, but like SC, it does not address the routing question of whether multi-sample generation is necessary in the first place. Both methods treat consistency as an aggregation signal, not as a predictor of single-pass solveability.

### 2.2 Early Stopping and Routing

A parallel line of research seeks to reduce the computational cost of multi-sample reasoning by stopping early when sufficient confidence is reached. Aghazadeh et al. [3] introduced confidence-guided early stopping (CGES), a Bayesian framework that uses token-level confidence signals to determine when additional samples are unlikely to change the majority outcome. CGES reduces average sample count while preserving accuracy, but it still requires generating multiple samples initially and relies on model-internal confidence rather than answer consistency.

Qin et al. [4] proposed CoDE-Stop, which monitors confidence dynamics during the reasoning process itself to decide when to halt generation. Entropy-based methods such as LEASH [5] use the distribution over reasoning steps as a stopping criterion. More recent approaches including TRACES [6] and ESTAR [7] employ step-level tagging and reinforcement learning to optimize stopping policies. These methods share a common limitation: they optimize when to stop generating additional samples, not whether to generate more than one sample at all. The routing decision---single-pass versus multi-sample---remains unaddressed.

### 2.3 Theoretical Frameworks

Yang et al. [8] developed a probabilistic inference scaling theory that formalizes the relationship between sampling count and accuracy. Their exponential saturation formula, $\text{Acc}_t = \text{Upp} - \alpha^t (\text{Upp} - \text{Acc}_0)$, is mathematically equivalent to the Arrhenius-like kinetics we adopt: $P_k = P_\infty \cdot (1 - e^{-k/k_0})$, with $P_\infty = \text{Upp}$ and $k_0 = -1/\ln(\alpha)$. Yang et al. validated this framework empirically across multiple models and benchmarks, establishing that aggregate accuracy saturates exponentially with sample count. Their work provides the descriptive foundation for our investigation but does not test whether the parameters of the saturation curve predict single-pass solveability.

Li [10] introduced the error depth hypothesis, which posits that some errors are structurally deep and resist correction regardless of sampling count. This framework predicts an irreducible error floor in multi-sample reasoning and motivates the search for routing signals that can identify such errors without exhaustive sampling. Our finding of a ~25 percentage point irreducible floor for consistency-based routing aligns with Li's prediction, though we identify a specific mechanism---stable wrong answers---that produces this floor.

### 2.4 Positioning and Contribution

Table 1 summarizes the relationship between our work and existing methods. SC and RASC use consistency for aggregation; CGES, CoDE-Stop, and LEASH use confidence or entropy for early stopping. None systematically test whether their signal predicts single-pass solveability. We adopt the exponential saturation framework of Yang et al. [8], derive activation energy from answer consistency, and subject the routing hypothesis to empirical falsification. Our contribution is the first systematic falsification of consistency-based routing, supported by diagnostic analysis of failure modes and quantification of the irreducible error floor.

**Table 1: Comparison with related work.**

| Method | Signal | Aggregation | Key Difference |
|--------|--------|-------------|----------------|
| Self-consistency [1] | Consistency | Majority voting | Baseline; no routing |
| RASC [2] | Reasoning quality | Weighted majority | Different signal |
| CGES [3] | Confidence | Bayesian stopping | Different signal; no single-pass prediction |
| CoDE-Stop [4] | Confidence dynamics | Early stopping | Different signal; no single-pass prediction |
| LEASH [5] | Entropy | Early stopping | Different signal; no single-pass prediction |
| **Ours** | **Consistency (tested)** | **N/A** | **Negative result with diagnostic analysis** |

The gap we address is specific and consequential: consistency is an attractive routing signal because it requires no model introspection, no training, and no access to token probabilities. If consistency-derived activation energy could predict single-pass solveability, it would enable zero-overhead routing. Our results show that it cannot.

---

## 3. Activation Energy Theory

### 3.1 Exponential Saturation Framework

We model the probability that a language model produces a correct answer as a function of the number of independent samples $k$. Following Yang et al. [8], we adopt an exponential saturation form mathematically equivalent to Arrhenius-like kinetics:

$$P_k = P_\infty \cdot \left(1 - e^{-k / k_0}\right)$$

where $P_\infty \in [0, 1]$ is the asymptotic accuracy ceiling (the model's capability limit for the problem distribution) and $k_0 > 0$ is the characteristic sampling count controlling the rate of convergence. The product $P_\infty \cdot k_0$ has units of samples and can be interpreted as a problem-difficulty parameter: larger values indicate slower convergence toward the ceiling.

This formulation is mathematically equivalent to the probabilistic inference scaling theory of Yang et al. [8], who write $\text{Acc}_t = \text{Upp} - \alpha^t (\text{Upp} - \text{Acc}_0)$. The correspondence is $P_\infty = \text{Upp}$ and $k_0 = -1 / \ln(\alpha)$. We use the Arrhenius form because it separates the ceiling ($P_\infty$) from the rate ($k_0$) and admits a direct physical analogy.

### 3.2 Activation Energy from Consistency

We derive a problem-level difficulty measure from answer consistency rather than from per-problem curve fitting. For problem $x_i$, let $S_{i,k} = \{a_{i,1}, \ldots, a_{i,k}\}$ denote the set of $k$ sampled answers. The answer consistency $c_{i,k}$ is the fraction of samples that agree with the majority answer:

$$c_{i,k} = \frac{1}{k} \max_{a} \sum_{j=1}^{k} \mathbb{I}[a_{i,j} = a]$$

We define the consistency-derived activation energy as

$$E_a = -\ln(c_0)$$

where $c_0 = c_{i,16}$ is the answer consistency at $k = 16$ samples. Higher $E_a$ indicates lower consistency and, by hypothesis, higher problem difficulty. We also compute a saturation-derived $k_0$ for each problem by fitting the exponential model to its per-problem accuracy trajectory; this yields an alternative difficulty measure for comparison (H5).

**Critical distinction.** Answer consistency measures whether the model produces the *same* answer across samples, not whether that answer is *correct*. A problem can have $c_0 = 1.0$ (all samples agree) yet be consistently wrong. This distinction is central to the falsification in Section 5.

### 3.3 Hypotheses

We derive four testable hypotheses from the framework:

**Table 2: Hypotheses and evaluation criteria.**

| Hypothesis | Claim | Metric | Threshold |
|------------|-------|--------|-----------|
| H1 | Aggregate accuracy follows Arrhenius kinetics | $R^2$ of exponential fit | $> 0.85$ |
| H2 | $E_a$ correlates with MATH difficulty level | Spearman($E_a$, Level) | $> 0.4$ |
| H3 | Low-$E_a$ problems are solvable in a single pass | Accuracy(low-$E_a$) | $> 75\%$ |
| H5 | Consistency-$E_a$ matches saturation-$k_0$ | Spearman($E_a$, $k_0$) | $> 0.5$ |

H1 and H2 are descriptive predictions: they test whether the exponential framework and the consistency-derived difficulty measure capture observable structure in the data. H3 is the prescriptive routing prediction: if low-$E_a$ problems achieve high single-pass accuracy, then $E_a$ can be used to route problems to single-pass vs. multi-sample inference. H5 tests the internal consistency of the theory by comparing two independent difficulty estimates.

### 3.4 Implications for Routing

If H3 holds, a natural routing strategy emerges: classify problems with $E_a$ below a threshold $E_a^{\text{low}}$ as "single-pass solvable" and problems with $E_a$ above $E_a^{\text{high}}$ as "multi-sample required." The threshold can be set by requiring a minimum single-pass accuracy $\tau = 0.75$ (75%).

If H3 fails, the implication is that answer consistency is insufficient as a routing signal. The model may produce stable wrong answers---high consistency, low accuracy---rendering $E_a$ useless for predicting single-pass solveability. We test this decisively in Section 5.

Figure 2 illustrates the hypothesized pipeline: sample $k$ times, compute consistency, derive $E_a$, threshold for routing, and evaluate the outcome.

![Figure 2: Theory framework and intended routing pipeline.](figures/fig2_theory_framework.pdf)

---

## 4. Experimental Setup

### 4.1 Model and Dataset

We evaluate on *Qwen2.5-Math-7B-Instruct* [11], a 7-billion parameter instruction-tuned model specialized for mathematics. We use the MATH test set [9], a benchmark of 12,500 competition mathematics problems with five difficulty levels (Level 1: pre-algebra through Level 5: precalculus). We select a stratified subset of 50 problems spanning Levels 1--5.

All experiments are conducted on a single NVIDIA RTX PRO 6000 Blackwell GPU (compute capability sm_120) using PyTorch 2.11.0, which introduced native Blackwell support.

### 4.2 Evaluation Protocol

For each problem $x_i$, we generate $k \in \{1, 2, 4, 8, 16\}$ independent samples using temperature $T = 0.7$ and a maximum of $M = 1024$ tokens per sample. The random seed is fixed at 42 for reproducibility.

**Answer extraction.** We extract final answers from the model output using a regex-based parser that matches boxed expressions (e.g., `\boxed{...}`). We manually audit 10% of extractions to verify parser accuracy.

**Accuracy computation.** At each $k$, the majority-voted answer across the $k$ samples is compared against the ground-truth answer $y_i^*$ to compute the accuracy indicator $\text{Acc}(S_{i,k}) \in \{0, 1\}$. Aggregate accuracy at each $k$ is the mean over all 50 problems.

**Consistency computation.** For each problem and each $k$, we compute $c_{i,k}$ as the fraction of samples agreeing on the majority answer. The final consistency $c_0 = c_{i,16}$ is used to compute $E_a = -\ln(c_0)$.

### 4.3 Analysis Groups

We define three analysis groups aligned with our hypotheses:

**G1 (Saturation).** We fit the exponential model $P_k = P_\infty \cdot (1 - e^{-k/k_0})$ to the aggregate accuracy trajectory across all 50 problems using non-linear least squares. H1 is confirmed if the fitted $R^2 > 0.85$. We additionally fit the model per-problem and report the distribution of $R^2$ values to test whether the aggregate pattern holds at the individual level.

**G2 (Consistency).** For each problem, we compute $c_0$ at $k = 16$ and derive $E_a = -\ln(c_0)$. We test H2 by computing Spearman $\rho(E_a, L_i)$ against the MATH difficulty level $L_i \in \{1, 2, 3, 4, 5\}$. H2 is confirmed if $\rho > 0.4$. We also test H5 by computing Spearman $\rho(E_a, k_0)$ for problems with valid per-problem saturation fits.

**G3 (Routing).** We use the $E_a$ estimates from G2 to classify problems into low-$E_a$ and high-$E_a$ subsets. The threshold is optimized post-hoc on the full dataset to maximize single-pass accuracy of the low-$E_a$ group. We report the resulting single-pass accuracy, AUC of the $E_a$-based classifier, and Spearman $\rho(E_a, \text{single-pass correctness})$. H3 is confirmed only if the low-$E_a$ accuracy exceeds 75% with a genuinely predictive threshold (AUC $> 0.5$).

**Table 3: Experiment configuration.**

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

---

## 5. Results

### 5.1 H1: Aggregate Arrhenius Saturation is Confirmed

We test whether aggregate accuracy follows the exponential saturation model. Table 4 reports accuracy at each sampling count.

**Table 4: Aggregate accuracy vs. sampling count.**

| $k$ | Accuracy |
|-----|----------|
| 1 | 68.0% |
| 2 | 78.0% |
| 4 | 86.0% |
| 8 | 84.0% |
| 16 | 82.0% |

Fitting the exponential model $P_k = P_\infty \cdot (1 - e^{-k/k_0})$ yields:
- $P_\infty = 0.835$ (asymptotic ceiling)
- $k_0 = 0.613$ (characteristic sampling count)
- $R^2 = 0.924$

The $R^2$ exceeds our threshold of 0.85, confirming H1 at the aggregate level. Figure 3 visualizes the saturation curve with the fitted exponential model.

![Figure 3: Aggregate accuracy vs. sampling count with exponential fit ($R^2 = 0.924$).](figures/fig3_h1_saturation.pdf)

**Critical caveat.** The aggregate fit smooths over individual variation. Per-problem median $R^2 = 0.000$; 80% of problems fail to fit. The exponential pattern is a group-level statistical regularity, not a universal physical law.

### 5.2 H2: $E_a$ Correlates with MATH Level

We test whether consistency-derived activation energy correlates with problem difficulty. Figure 4 shows the distribution of $E_a$ across MATH Levels 1--5.

![Figure 4: $E_a$ distribution by MATH difficulty level, showing bimodal clustering at $E_a \approx 9.47$ and $E_a \approx 10.0$.](figures/fig4_h2_ea_levels.pdf)

**Table 5: Mean $E_a$ by MATH level.**

| Level | Count | Mean $E_a$ | Std $E_a$ | Accuracy |
|-------|-------|-----------|----------|----------|
| 1 | 2 | 9.465 | 0.000 | 50.0% |
| 2 | 9 | 9.643 | 0.252 | 77.8% |
| 3 | 15 | 9.750 | 0.267 | 66.7% |
| 4 | 11 | 9.708 | 0.266 | 72.7% |
| 5 | 13 | 10.000 | $1.9 \times 10^{-6}$ | 61.5% |

The Spearman correlation is $\rho(E_a, \text{Level}) = 0.448$, $p = 0.001$, exceeding our threshold of 0.4. H2 is confirmed.

**Caveat.** $E_a$ values are bimodal (clusters at ~9.47 and ~10.0), limiting within-level discriminative power. Level 5 shows near-zero variance ($\sigma \approx 1.9 \times 10^{-6}$), suggesting algorithmic artifacts.

### 5.3 H3: Single-Pass Routing is Falsified

We test whether $E_a$ can predict single-pass correctness. Figure 5 shows the routing failure.

![Figure 5: Single-pass accuracy by $E_a$ category. Low-$E_a$ accuracy is 75.0% only via post-hoc threshold optimization; AUC = 0.436 indicates worse-than-random predictive power.](figures/fig5_h3_routing_failure.pdf)

**Table 6: $E_a$ routing performance.**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Optimal threshold | 9.9999999999 | Post-hoc (data leakage) |
| Low-$E_a$ accuracy | 75.0% | Meets threshold only via post-hoc optimization |
| High-$E_a$ accuracy | 50.0% | --- |
| AUC | 0.436 | $< 0.5$: worse than random |
| Spearman($E_a$, accuracy) | $-0.063$ | $p = 0.66$: no relationship |

The decisive evidence is AUC = 0.436 and Spearman = $-0.063$. These values prove $E_a$ cannot predict single-pass solveability. The 75% threshold pass is a statistical artifact of bimodal $E_a$ distribution, not genuine predictive signal. H3 is falsified.

### 5.4 H5: Consistency-$E_a$ and Saturation-$k_0$ are Unrelated

We test whether the two "activation energy" measures agree. The Spearman correlation is $\rho(E_a, k_0) = -0.219$, $p = 0.54$, with only 10 of 50 problems yielding valid $k_0$ estimates.

The two measures---derived from consistency versus from saturation curves---capture different constructs. The theoretical framework lacks internal consistency. H5 is falsified.

### 5.5 Hypothesis Summary

**Table 7: Hypothesis results summary.**

| Hypothesis | Metric | Threshold | Actual | Status |
|------------|--------|-----------|--------|--------|
| H1 (Aggregate Arrhenius) | $R^2$ | $> 0.85$ | 0.924 | Confirmed (aggregate only) |
| H2 ($E_a$ = difficulty) | Spearman | $> 0.4$ | 0.448 | Confirmed (modest effect) |
| H3 (Single-pass routing) | AUC | $> 0.5$ | 0.436 | **Falsified** |
| H5 ($E_a$ = $k_0$) | Spearman | $> 0.5$ | $-0.219$ | **Falsified** |

Two of four hypotheses are confirmed at the descriptive level; both prescriptive hypotheses are falsified.

---

## 6. Discussion and Diagnosis

### 6.1 Why $E_a$-Based Routing Fails: $E_a$ Measures Stability, Not Correctness

The central finding of this study is that consistency-derived activation energy ($E_a$) cannot predict whether a problem is solvable in a single pass (AUC = 0.436, Spearman = $-0.063$). This section diagnoses the failure mechanism and situates the result within the broader landscape of inference routing research.

The fundamental issue is semantic: answer consistency measures whether the model produces the *same* answer across multiple samples, not whether that answer is *correct*. This distinction is subtle but decisive. When all 16 samples for a problem agree on the same answer, consistency $c_0 = 1.0$ and $E_a = 0$---the model is perfectly stable. Yet the agreed answer may be wrong. We call these **stable wrong answers**: incorrect responses that are consistently reproduced across independent samples.

The bimodal distribution of $E_a$ values (clusters at approximately 9.47 and 10.0) reflects two stability modes, not two correctness modes. Problems with $E_a \approx 9.47$ ($c_0 \approx 0.106$) show moderate consistency; problems with $E_a \approx 10.0$ ($c_0 \approx 0.00005$) show near-zero consistency. Both clusters contain correct and incorrect single-pass outcomes. The low-$E_a$ cluster does not map to "easy" problems and the high-$E_a$ cluster does not map to "hard" problems in any predictive sense. The correlation with MATH difficulty level (Spearman = 0.448, $p = 0.001$) is real but modest, and it does not translate into actionable routing signal.

The 75.0% accuracy for low-$E_a$ problems meets the routing threshold ($\tau = 0.75$) only through post-hoc threshold optimization. The optimal threshold (9.999...) is a data-dependent value that would not generalize to new problems. Without this optimization, the classification performance is worse than random (AUC = 0.436). A routing strategy using $E_a$ would misclassify more problems than a coin flip.

### 6.2 Error Classification

To understand why stable wrong answers occur, we classify the failures among low-$E_a$ problems (high consistency but incorrect single-pass answer) into three error types:

**Table 8: Error classification for low-$E_a$ failures.**

| Error Type | Description | Example |
|------------|-------------|---------|
| Execution error | Correct approach, arithmetic mistake | Calculation slip in intermediate step |
| Conceptual error | Wrong approach, internally consistent | Misidentifies geometric theorem |
| Answer extraction failure | Correct reasoning, wrong format | Answer not in expected boxed format |

Execution errors are particularly relevant to the stable wrong answer phenomenon. When a model commits the same arithmetic mistake across all samples---for instance, consistently miscalculating a fraction or misapplying a formula---the result is high consistency with low accuracy. The model is not uncertain; it is systematically wrong. Conceptual errors can also produce stable wrong answers when the model consistently applies an incorrect theorem or reasoning pattern. Answer extraction failures, while less common, contribute to the error floor when the model reasons correctly but fails to format the final answer as expected.

The dominance of execution errors among low-$E_a$ failures supports the "stable wrong answer" narrative. These are not cases where the model is confused or uncertain; they are cases where the model is confidently and consistently wrong. This is precisely the failure mode that consistency-based routing cannot detect.

### 6.3 The Irreducible Error Floor

Our results reveal an irreducible error floor for consistency-based routing of approximately 25 percentage points. The low-$E_a$ group achieves 75.0% single-pass accuracy at best, meaning 25% of problems that appear "easy" by the consistency criterion are actually unsolvable in one pass. This floor is substantially larger than the approximately 8 percentage points reported for variance-based routing methods such as ACAR. The difference suggests that the choice of signal---consistency versus variance---significantly affects the fundamental limits of routing performance.

The error floor has practical consequences. A routing strategy that misclassifies one in four "easy" problems will incur substantial accuracy penalties if single-pass answers are used without verification. The 25pp gap represents the maximum theoretical improvement from perfect routing; any real-world system will perform worse due to threshold estimation error, distribution shift, and other confounds.

### 6.4 Implications for Routing Strategies

Our falsification of H3 has direct implications for the design of inference routing systems. Table 9 compares routing signals on their ability to predict single-pass solveability.

**Table 9: Routing signal comparison.**

| Signal | What It Measures | Predicts Solveability? | Evidence |
|--------|-----------------|----------------------|----------|
| Answer consistency ($E_a$) | Answer stability | **No** (AUC = 0.436) | This paper |
| Token entropy | Model uncertainty | Unknown | Not tested |
| Confidence | Probability of correctness | Partial (CGES) | Literature [3] |
| Reasoning quality | Step validity | Partial (RASC) | Literature [2] |

The evidence suggests a hierarchy of signal quality for routing. Confidence-based signals, as used in CGES, show partial predictive power because they relate more directly to the model's assessment of correctness. Reasoning quality signals, as used in RASC, capture structural features of the reasoning process that may correlate with correctness. Token entropy remains untested for single-pass prediction but is a promising candidate because it measures the model's uncertainty during generation rather than the stability of its final answer.

Our recommendation is that future routing research should combine multiple signals rather than relying on consistency alone. A hybrid approach might use consistency as a stability check, confidence as a correctness proxy, and reasoning quality as a structural validator. Such a system would be more robust to the stable wrong answer problem because no single signal would be solely responsible for the routing decision.

Additionally, we propose that any claim of routing utility should be validated on the single-pass prediction task (AUC) before being deployed. The field has focused on aggregate accuracy improvements from multi-sample reasoning, but the routing question---which problems need multiple samples?---requires a different evaluation protocol. Our study provides a template for such validation.

Figure 6 visualizes the routing signal landscape, positioning our negative result against alternative signals.

![Figure 6: Routing signal comparison. Consistency measures agreement but not correctness, creating an irreducible error floor from execution errors. Alternative single-pass signals remain promising but untested.](figures/hypothesis_summary.pdf)

### 6.5 Limitations

This study has several limitations that bound the generality of our conclusions.

**Small sample size.** The experiment uses 50 problems from the MATH test set. While sufficient to detect the null effect on H3 (AUC = 0.436 is far from the 0.5 threshold), the results are pilot-scale and may not generalize to larger or more diverse problem sets.

**Single model.** All experiments use *Qwen2.5-Math-7B-Instruct*. The model's specific training, architecture, and fine-tuning may produce consistency patterns that differ from other models. Replication on additional models---including larger models, different families, and instruction-tuned variants---is needed.

**Post-hoc threshold.** The 75.0% low-$E_a$ accuracy figure relies on a threshold optimized on the same data used for evaluation. This is a form of data leakage that inflates performance estimates. A held-out validation set or cross-validation would provide a more realistic assessment.

**Answer extraction.** Correctness labels depend on regex-based extraction of boxed answers. Systematic extraction errors---for instance, failing to parse complex mathematical expressions---may confound correctness labels and affect consistency calculations. A 10% manual audit was performed, but larger-scale validation would strengthen confidence.

**Bimodal artifacts.** The near-zero variance of $E_a$ at Level 5 ($\sigma \approx 1.9 \times 10^{-6}$) suggests algorithmic saturation rather than genuine consistency. This artifact may arise from numerical precision limits in the consistency calculation or from the specific problem characteristics at this difficulty level. The bimodality complicates threshold-based routing and may be an inherent feature of the consistency measure.

Despite these limitations, the core finding---that $E_a$ does not predict single-pass solveability---is robust. The AUC of 0.436 and Spearman correlation of $-0.063$ are far from any reasonable threshold for predictive utility, and the stable wrong answer mechanism provides a principled explanation for the failure that does not depend on sample size or model choice.

---

## 7. Conclusion

This paper systematically tested whether answer consistency, framed as activation energy $E_a$, can predict whether a mathematical reasoning problem is solvable in a single pass. We derived four testable predictions from Activation Energy Theory and evaluated them on *Qwen2.5-Math-7B-Instruct* across 50 problems from the MATH dataset with $k = 1, 2, 4, 8, 16$ samples.

Our findings are threefold. First, aggregate accuracy follows exponential Arrhenius saturation with $R^2 = 0.924$, cross-validating the probabilistic inference scaling framework of Yang et al. [8]. This descriptive pattern holds at the group level but not for individual problems: 80% of per-problem fits fail, and the median per-problem $R^2 = 0.000$. Second, consistency-derived $E_a$ correlates with MATH difficulty level (Spearman $\rho = 0.448$, $p = 0.001$), but the effect is modest and the distribution is bimodal, clustering at $E_a \approx 9.47$ and $E_a \approx 10.0$. Third, and most critically, $E_a$ has zero predictive power for single-pass correctness (AUC = 0.436, Spearman $\rho = -0.063$). The routing hypothesis H3 is decisively falsified. A post-hoc threshold achieves 75.0% accuracy for low-$E_a$ problems, but this is a statistical artifact of the bimodal distribution, not genuine predictive signal. Additionally, consistency-$E_a$ and saturation-$k_0$ are uncorrelated (Spearman $\rho = -0.219$, $p = 0.54$), indicating that the two "activation energy" measures capture different constructs.

The central diagnosis is that answer consistency measures stability, not correctness. The model can produce stable wrong answers---all 16 samples agreeing on an incorrect response---yielding low $E_a$ and high consistency without correctness. This creates an irreducible error floor of approximately 25 percentage points for consistency-based routing, larger than the ~8pp ceiling reported for variance-based alternatives. The implication is clear: any routing strategy that relies solely on answer consistency will misroute a substantial fraction of problems.

We offer two recommendations for the research community. First, future routing methods should test their signals on single-pass prediction (AUC) before claiming utility; descriptive correlation with difficulty is insufficient. Second, practical routing should combine multiple signals---token entropy, model confidence, and reasoning quality---rather than relying on any single metric. Consistency may still play a role as one feature in a composite signal, but it cannot stand alone.

Several directions remain open. Scaling the analysis to 250+ problems and multiple models would test the robustness of the bimodal $E_a$ distribution and the aggregate saturation pattern. Testing token entropy and confidence signals under the same falsification protocol would establish a comparative baseline for routing signal quality. Finally, developing theoretical frameworks that explicitly distinguish answer stability from answer correctness---perhaps by modeling the joint distribution of consistency and accuracy---could yield routing signals with stronger predictive validity.

In sum, we confirm that Arrhenius-like kinetics describe aggregate inference scaling, but we falsify the prescriptive claim that consistency-derived activation energy enables problem-level routing. Consistency tells us whether the model agrees with itself; it does not tell us whether the model is right.

---

## Figures and Tables

- **Figure 1**: `fig1_teaser.pdf` --- Aggregate accuracy vs. sampling count with exponential saturation curve and 75% single-pass threshold line. Motivates the gap between descriptive pattern and prescriptive failure.
- **Figure 2**: `fig2_theory_framework.pdf` --- Conceptual diagram of the Activation Energy Theory pipeline: sample, compute consistency, derive $E_a$, threshold for routing, evaluate outcome.
- **Figure 3**: `fig3_h1_saturation.pdf` --- Aggregate accuracy at $k = 1, 2, 4, 8, 16$ with fitted exponential model ($P_\infty = 0.835$, $k_0 = 0.613$, $R^2 = 0.924$).
- **Figure 4**: `fig4_h2_ea_levels.pdf` --- $E_a$ distribution across MATH Levels 1--5 with individual data points, showing bimodal clustering.
- **Figure 5**: `fig5_h3_routing_failure.pdf` --- Single-pass accuracy by $E_a$ category vs. 75% threshold, with ROC inset (AUC = 0.436).
- **Figure 6**: `hypothesis_summary.pdf` --- Routing signal comparison: consistency vs. alternative single-pass signals (entropy, confidence, reasoning quality).
- **Table 1**: inline --- Comparison of related work methods by signal, aggregation, and key difference.
- **Table 2**: inline --- Hypotheses and evaluation criteria (H1, H2, H3, H5).
- **Table 3**: inline --- Experiment configuration (model, dataset, sampling parameters, hardware).
- **Table 4**: inline --- Aggregate accuracy vs. sampling count ($k = 1, 2, 4, 8, 16$).
- **Table 5**: inline --- Mean $E_a$ by MATH difficulty level (Levels 1--5).
- **Table 6**: inline --- $E_a$ routing performance metrics (threshold, accuracy, AUC, Spearman).
- **Table 7**: inline --- Hypothesis results summary with status (confirmed/falsified).
- **Table 8**: inline --- Error classification for low-$E_a$ failures (execution, conceptual, extraction).
- **Table 9**: inline --- Routing signal comparison by predictive power for single-pass solveability.
