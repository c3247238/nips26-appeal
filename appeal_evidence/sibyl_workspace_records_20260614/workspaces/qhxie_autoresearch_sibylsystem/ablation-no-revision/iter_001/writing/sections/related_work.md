# Related Work

## Multi-Sample Reasoning

The dominant approach to improving LLM reasoning accuracy without additional training is to generate multiple independent samples and aggregate their answers. Wang et al. [1] introduced self-consistency (SC), which samples $k$ chain-of-thought (CoT) reasoning paths and selects the answer via majority voting. SC established that answer consistency across samples is a reliable signal for selecting the most probable correct answer at the aggregate level. However, SC requires generating all $k$ samples upfront, making it computationally expensive even for problems that might be solvable in a single pass.

Wan et al. [2] proposed reasoning-aware self-consistency (RASC), which extends SC by weighting votes according to reasoning quality signals rather than treating all samples equally. RASC demonstrated that reasoning-quality-weighted aggregation outperforms uniform majority voting, but like SC, it does not address the routing question of whether multi-sample generation is necessary in the first place. Both methods treat consistency as an aggregation signal, not as a predictor of single-pass solveability.

## Early Stopping and Routing

A parallel line of research seeks to reduce the computational cost of multi-sample reasoning by stopping early when sufficient confidence is reached. Aghazadeh et al. [3] introduced confidence-guided early stopping (CGES), a Bayesian framework that uses token-level confidence signals to determine when additional samples are unlikely to change the majority outcome. CGES reduces average sample count while preserving accuracy, but it still requires generating multiple samples initially and relies on model-internal confidence rather than answer consistency.

Qin et al. [4] proposed CoDE-Stop, which monitors confidence dynamics during the reasoning process itself to decide when to halt generation. Entropy-based methods such as LEASH [5] use the distribution over reasoning steps as a stopping criterion. More recent approaches including TRACES [6] and ESTAR [7] employ step-level tagging and reinforcement learning to optimize stopping policies. These methods share a common limitation: they optimize when to stop generating additional samples, not whether to generate more than one sample at all. The routing decision---single-pass versus multi-sample---remains unaddressed.

## Theoretical Frameworks

Yang et al. [8] developed a probabilistic inference scaling theory that formalizes the relationship between sampling count and accuracy. Their exponential saturation formula, $\text{Acc}_t = \text{Upp} - \alpha^t (\text{Upp} - \text{Acc}_0)$, is mathematically equivalent to the Arrhenius-like kinetics we adopt: $P_k = P_\infty \cdot (1 - e^{-k/k_0})$, with $P_\infty = \text{Upp}$ and $k_0 = -1/\ln(\alpha)$. Yang et al. validated this framework empirically across multiple models and benchmarks, establishing that aggregate accuracy saturates exponentially with sample count. Their work provides the descriptive foundation for our investigation but does not test whether the parameters of the saturation curve predict single-pass solveability.

Li [9] introduced the error depth hypothesis, which posits that some errors are structurally deep and resist correction regardless of sampling count. This framework predicts an irreducible error floor in multi-sample reasoning and motivates the search for routing signals that can identify such errors without exhaustive sampling. Our finding of a ~25 percentage point irreducible floor for consistency-based routing aligns with Li's prediction, though we identify a specific mechanism---stable wrong answers---that produces this floor.

## Positioning and Contribution

Table 3 summarizes the relationship between our work and existing methods. SC and RASC use consistency for aggregation; CGES, CoDE-Stop, and LEASH use confidence or entropy for early stopping. None systematically test whether their signal predicts single-pass solveability. We adopt the exponential saturation framework of Yang et al. [8], derive activation energy from answer consistency, and subject the routing hypothesis to empirical falsification. Our contribution is the first systematic falsification of consistency-based routing, supported by diagnostic analysis of failure modes and quantification of the irreducible error floor.

| Method | Signal | Aggregation | Key Difference |
|--------|--------|-------------|----------------|
| Self-consistency [1] | Consistency | Majority voting | Baseline; no routing |
| RASC [2] | Reasoning quality | Weighted majority | Different signal |
| CGES [3] | Confidence | Bayesian stopping | Different signal; no single-pass prediction |
| CoDE-Stop [4] | Confidence dynamics | Early stopping | Different signal; no single-pass prediction |
| LEASH [5] | Entropy | Early stopping | Different signal; no single-pass prediction |
| **Ours** | **Consistency (tested)** | **N/A** | **Negative result with diagnostic analysis** |

The gap we address is specific and consequential: consistency is an attractive routing signal because it requires no model introspection, no training, and no access to token probabilities. If consistency-derived activation energy could predict single-pass solveability, it would enable zero-overhead routing. Our results show that it cannot.
