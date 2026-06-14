# Glossary: The Limits of Consistency-Based Activation Energy for Problem-Level Routing

## Core Concepts

### Activation Energy ($E_a$)
**Definition**: A scalar measure derived from answer consistency, defined as $E_a = -\ln(c_0)$ where $c_0$ is the fraction of samples agreeing on the majority answer at $k = 16$.
**Preferred**: "activation energy" or "$E_a$"
**Avoid**: "energy barrier" (chemistry-specific); "difficulty metric" (overstates predictive power)
**Critical note**: Measures answer stability, NOT correctness. High consistency can occur with consistently wrong answers.

### Aggregate Statistical Pattern
**Definition**: A mathematical relationship that holds for group-averaged data but not for individual instances.
**Preferred**: "aggregate pattern", "group-level regularity"
**Avoid**: "universal law", "physical law"
**Context**: The exponential saturation with $R^2 = 0.924$ is an aggregate pattern; 80% of individual problems fail to fit.

### Answer Consistency
**Definition**: The fraction of multiple samples that agree on the same (majority) answer.
**Preferred**: "answer consistency", "consistency"
**Avoid**: "confidence" (confused with probability-based confidence); "agreement" (ambiguous)
**Critical note**: Measures agreement among samples, not agreement with ground truth.

### Arrhenius Kinetics (in LLM reasoning)
**Definition**: The mathematical relationship $P_k = P_\infty \cdot (1 - e^{-k/k_0})$ describing how aggregate accuracy improves with sampling count.
**Preferred**: "exponential saturation", "Arrhenius-like kinetics"
**Avoid**: "Arrhenius equation" (chemistry-specific); "Arrhenius law" (overstates universality)
**Context**: Mathematically equivalent to Yang et al. (2025); describes group-average behavior only.

### Asymptotic Ceiling ($P_\infty$)
**Definition**: The maximum achievable accuracy regardless of sampling count, representing the model's capability limit for a given problem set.
**Preferred**: "asymptotic ceiling", "asymptotic accuracy"
**Avoid**: "upper bound" (reserved for theoretical bounds); "maximum accuracy" (ambiguous)

### Characteristic Sampling Count ($k_0$)
**Definition**: The parameter controlling the rate of convergence to $P_\infty$ in the exponential saturation model.
**Preferred**: "characteristic sampling count", "characteristic count"
**Avoid**: "characteristic time" (chemistry-specific)

### Consistency-Derived $E_a$
**Definition**: Activation energy estimated from answer consistency trajectories across samples.
**Preferred**: "consistency-derived $E_a$", "consistency-$E_a$"
**Avoid**: "estimated $E_a$" (ambiguous -- could mean from saturation curves)

### Saturation-Derived $k_0$
**Definition**: Characteristic count estimated by fitting the exponential model to per-problem accuracy curves.
**Preferred**: "saturation-derived $k_0$", "saturation-$k_0$"
**Avoid**: "fitted $k_0$" (ambiguous)

## Methods and Techniques

### Self-Consistency (SC)
**Definition**: A method where multiple independent samples are generated and the majority-voted answer is selected.
**Reference**: Wang et al. (2022)
**Preferred**: "self-consistency", "majority voting"

### Reasoning-Aware Self-Consistency (RASC)
**Definition**: A method combining early stopping with weighted majority voting based on reasoning quality signals.
**Reference**: Wan et al. (2024, arXiv:2408.17017)
**Preferred**: "RASC"

### Confidence-Guided Early Stopping (CGES)
**Definition**: A Bayesian framework for early stopping in multi-sample reasoning using token-level confidence signals.
**Reference**: Aghazadeh et al. (2025, arXiv:2511.02603)
**Preferred**: "CGES"

### CoDE-Stop
**Definition**: A method using confidence dynamics (how confidence evolves during the reasoning process) for early stopping.
**Reference**: Qin et al. (2025)
**Preferred**: "CoDE-Stop"

### Multi-Sample Reasoning
**Definition**: Generating multiple independent samples and aggregating results to improve accuracy.
**Preferred**: "multi-sample reasoning", "multi-sampling"
**Avoid**: "beam search", "nucleus sampling" (specific sampling strategies)

### Single-Pass Reasoning
**Definition**: Generating only one sample for a given problem, without aggregation.
**Preferred**: "single-pass", "single-sample"
**Avoid**: "greedy decoding" (specific strategy)

### Early Stopping
**Definition**: Stopping sample generation before reaching a fixed number when sufficient confidence or consistency is reached.
**Preferred**: "early stopping"
**Avoid**: "adaptive sampling" (overlapping but distinct)

### Routing
**Definition**: The decision of which inference strategy (single-pass, multi-sample, tools) to apply to a given problem.
**Preferred**: "routing", "inference routing"
**Avoid**: "scheduling" (computation scheduling is different)

## Error Types

### Execution Error
**Definition**: A mistake in calculation or arithmetic that does not affect the overall reasoning approach.
**Preferred**: "execution error", "calculation error"
**Avoid**: "slip error" (psychology term)
**Context**: Can lead to high consistency (all samples make the same mistake) but incorrect answers.

### Conceptual Error
**Definition**: A fundamental misunderstanding of the problem or approach that affects all reasoning attempts.
**Preferred**: "conceptual error", "reasoning error"
**Avoid**: "knowledge error" (ambiguous)

### Answer Extraction Failure
**Definition**: Correct reasoning but failure to extract the final answer in the expected format.
**Preferred**: "answer extraction failure", "format error"
**Avoid**: "parsing error" (technical term with different meaning)

### Stable Wrong Answer
**Definition**: An incorrect answer that is consistently produced across multiple samples, yielding high consistency but low accuracy.
**Preferred**: "stable wrong answer", "consistently wrong answer"
**Avoid**: "false consensus" (social science term)
**Context**: The central failure mode of consistency-based routing.

## Datasets

### MATH
**Definition**: A dataset of 12,500 mathematics competition problems with 5 difficulty levels (Level 1: pre-algebra to Level 5: precalculus).
**Reference**: Hendrycks et al. (2021)
**Preferred**: "MATH", "MATH dataset"
**Note**: When citing specific levels, use "Level 1" through "Level 5"

### GSM8K
**Definition**: A dataset of 8,500 middle school mathematics word problems.
**Reference**: Cobbe et al. (2021)
**Preferred**: "GSM8K"

## Models

### Qwen2.5-Math-7B-Instruct
**Definition**: A 7-billion parameter instruction-tuned language model specialized for mathematics.
**Reference**: Qwen Team (2024)
**Preferred**: "*Qwen2.5-Math-7B-Instruct*" (first mention), "the model" (subsequent)

### DeepSeek-Math-7B
**Definition**: A 7-billion parameter mathematics-specialized model.
**Reference**: DeepSeek-AI (2024)
**Preferred**: "DeepSeek-Math-7B"

## Statistical Terms

### Spearman Correlation
**Definition**: A rank-based correlation coefficient measuring monotonic relationships between two variables.
**Preferred**: "Spearman correlation", "Spearman's rho"
**Symbol**: $\rho$

### Coefficient of Determination ($R^2$)
**Definition**: The proportion of variance in the dependent variable explained by the model.
**Preferred**: "$R^2$", "R-squared"
**Note**: Use "$R^2$" not "R squared"

### Area Under the ROC Curve (AUC)
**Definition**: A measure of classification performance; 0.5 indicates random guessing, 1.0 indicates perfect classification.
**Preferred**: "AUC", "area under the ROC curve"
**Note**: AUC < 0.5 means the predictor is worse than random (reversing predictions would help).

### Post-Hoc Threshold
**Definition**: A decision threshold optimized on the same data used for evaluation, leading to optimistic performance estimates.
**Preferred**: "post-hoc threshold", "data-dependent threshold"
**Avoid**: "optimized threshold" (ambiguous)
**Context**: The 75% low-$E_a$ accuracy relies on a post-hoc threshold, making it unreliable.

### Bimodal Distribution
**Definition**: A probability distribution with two distinct peaks (modes).
**Preferred**: "bimodal distribution", "bimodal clustering"
**Context**: $E_a$ values cluster at ~9.47 and ~10.0, creating a bimodal distribution that complicates threshold-based routing.

## Hypothesis Status Terms

### Confirmed
**Definition**: The data supports the hypothesis at the specified threshold.
**Preferred**: "confirmed"
**Avoid**: "proven", "validated" (too strong)

### Falsified
**Definition**: The data contradicts the hypothesis at the specified threshold.
**Preferred**: "falsified"
**Avoid**: "disproven", "refuted" (philosophical distinctions)

### Borderline
**Definition**: The data is near the threshold but does not clearly confirm or falsify.
**Preferred**: "borderline"

## Writing Conventions

### Numbers
- Spell out numbers one through nine; use numerals for 10 and above
- Use numerals with units: "50 problems", "16 samples"
- Use numerals with statistics: "$R^2 = 0.924$", "$p = 0.001$"

### Percentages
- Format: "68.0%" (not "68.0 percent")
- Use "percentage points" (pp) for differences: "25pp gap"

### Hypotheses
- Format: "H1", "H2", "H3", "H5" (not "Hypothesis 1")
- Status: "confirmed", "falsified" (not "proven", "disproven")

### Model Names
- Italicize on first mention: *Qwen2.5-Math-7B-Instruct*
- Use "the model" in running text after first mention

### Abbreviations
- Define on first use: "large language models (LLMs)"
- Use consistently throughout

## Preferred Phrasing

| Avoid | Prefer |
|-------|--------|
| "groundbreaking", "revolutionary" | specific quantitative claims |
| "to the best of our knowledge" | state claims factually or omit |
| "significantly improves" | "improves by X percentage points" |
| "novel" | specific contribution descriptions |
| "in recent years", "in this work" | direct statements |
| "Furthermore", "Moreover" | "Additionally", "Also" or restructure |
| "It is worth noting that" | restructure or remove |
| "Ea predicts correctness" | "Ea does not predict correctness (AUC = 0.436)" |
| "the theory is proven" | "the aggregate pattern is confirmed" |
| "routing works" | "routing fails (AUC < 0.5)" |

## Citation Format
- Use numerical citations [1], [2] in running text
- Group citations: [1, 2, 3] for three sequential references
- Use semicolons for non-sequential: [1; 3; 5]
