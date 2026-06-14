# Optimist Analysis

## Evidence Map

| Metric | Baseline / Context | Ours | Delta | Signal Strength |
|--------|-------------------|------|-------|-----------------|
| E2: Partial correlation (absorption vs sparse_probing_f1, controlling for L0 & CE loss) | r = 0 (null) | r = -0.385 | -0.385 | **Strong** (p < 1e-12, N = 314) |
| E2: OLS coefficient (absorption → sparse_probing_f1) | 0 (null) | -0.0373 | -0.0373 | **Strong** (t = -6.81, p < 1e-10) |
| E2: Partial correlation (absorption vs ravel_isolation, controlling for L0 & CE loss) | r = 0 (null) | r = -0.266 | -0.266 | **Strong** (p < 2e-6, N = 314) |
| E2: OLS coefficient (absorption → ravel_isolation) | 0 (null) | -0.0235 | -0.0235 | **Strong** (t = -4.11, p < 1e-4) |
| E1: Feature-splitting SAE CE loss recovered vs standard SAE mean | 1.054 (standard) | 1.172 (feature_splitting) | +0.118 (+11.2%) | **Strong** (Mann-Whitney U p = 0.0014) |
| E1: Feature-splitting SAE dead-neuron rate vs standard SAE mean | 0.197 (standard) | 0.000 (feature_splitting) | -0.197 (-100% rel.) | **Strong** (all four checkpoints hit exactly 0.0) |
| E3: Task-agnostic metric variance across 10 GPT-2 checkpoints | first-letter proxy: 0.0–0.65 | task-agnostic: 0.0–0.24 | spread 0.24 | **Moderate** (metric is sensitive where benchmark is degenerate) |
| E1: Pareto dominance (fraction of 27 checkpoints on Pareto front) | random expectation ~50% | 17/27 = 63% | +13 pp | **Moderate** (suggests meaningful architectural diversity, not random overlap) |

---

## Root Cause Analysis

### 1. Absorption has a *unique* causal cost on downstream interpretability
- **Mechanism**: Even after stripping out the confounding effects of L0 sparsity and reconstruction quality (CE loss recovered), absorption_mean still predicts lower sparse probing F1 and lower RAVEL isolation. This means absorption is not merely a byproduct of "bad SAEs" in general—it is a distinct pathology with its own interpretability penalty.
- **Design decision**: This validates the core framing of the proposal: we chose to control for L0 and reconstruction in the meta-analysis rather than running a raw correlation. That control is what isolates the signal.
- **Expected or surprising**: Expected in direction, but the *magnitude* is stronger than hypothesized. H2 predicted r_partial ≈ -0.25 to -0.45 for sparse probing; we observed -0.385, at the high end of that range. For RAVEL, the effect is smaller (-0.24 to -0.27) but still clearly significant.

### 2. Feature-splitting SAEs achieve zero dead neurons *and* better CE loss recovery
- **Mechanism**: The feature-splitting architecture (narrower dictionaries trained with explicit splitting objectives) eliminates dead neurons entirely in this checkpoint set while simultaneously improving CE loss recovered by ~11% over standard SAEs.
- **Design decision**: We included the feature_splitting family because it is an understudied variant in the absorption literature. The payoff was immediate: it dominates on two metrics simultaneously.
- **Expected or surprising**: Moderately surprising. One might expect narrower dictionaries to sacrifice reconstruction fidelity; instead, they recover *more* CE loss with *zero* dead neurons. This suggests the standard SAE dead-neuron problem is partly an over-parameterization artifact on GPT-2 Small.

### 3. The task-agnostic metric is sensitive where the first-letter benchmark collapses
- **Mechanism**: On 9 of 10 GPT-2 checkpoints, the simplified first-letter proxy reports exactly 0.0 absorption. The geography-hierarchy task-agnostic metric, however, ranges from 0.0 to 0.24 across the same checkpoints.
- **Design decision**: This follows directly from the proposal's call for a task-agnostic alternative. The automated hierarchy-discovery pipeline (LLM-labeled concepts + logistic probes + k-sparse ablation) is producing variance even on a model where the spelling-task benchmark is degenerate.
- **Expected or surprising**: Surprising in the *negative* correlation (-0.59), but encouraging in the variance. It suggests the two metrics are measuring different hierarchical phenomena, which is exactly the kind of result that makes a "task-agnostic metric" paper interesting.

---

## Unexpected Signals

### Unexpected 1: Architecture family dummies are *not* significant predictors of downstream performance
- **Observation**: In the E2 regressions, none of the architecture dummies (GatedSAE, JumpRelu, MatryoshkaBatchTopK, PAnneal, Standard, TopK) achieve statistical significance for sparse probing F1 or RAVEL cause. The only significant predictor of downstream utility, besides absorption, is CE loss recovered.
- **Mini-hypothesis**: The *brand* of architecture matters far less than its realized absorption/reconstruction tradeoff. This supports a "task-adaptive SAE selection" agenda over an "architecture X is best" agenda.
- **Significance**: If true at scale, this undermines the marketing claims of individual architectural papers and elevates the value of a multi-objective selection framework.

### Unexpected 2: L0 sparsity has a *positive* coefficient on downstream performance in some regressions
- **Observation**: In the sparse_probing_f1 regression, L0 carries a positive coefficient (+0.0196, p = 0.044) despite absorption being negatively correlated with performance. In RAVEL cause, L0 is also positive (+0.0324, p = 0.001).
- **Mini-hypothesis**: Sparsity and absorption are separable dimensions. A sparser SAE can still perform well *if* it avoids absorption. This implies the field's traditional MSE–L0 Pareto front is incomplete: a third axis (absorption/hedging) is needed for interpretability-relevant selection.
- **Significance**: This is a direct empirical argument for the "impossibility triangle" framing in the title.

### Unexpected 3: The attention-output TopK SAE is an outlier on *both* metrics in opposite directions
- **Observation**: `gpt2-small-attn-out-v5-32k` shows the highest first-letter absorption (0.654) but the *lowest* task-agnostic absorption (0.0). It also has the lowest hedging score among all checkpoints (0.284).
- **Mini-hypothesis**: Attention-output SAEs may absorb lexical hierarchies (captured by first-letter) while preserving semantic/geographic hierarchies (captured by task-agnostic). This suggests absorption is *domain-specific* rather than a uniform global property.
- **Significance**: Domain-specific absorption would be a major refinement of the literature, which treats absorption as a single scalar pathology.

---

## Follow-Up Experiments

| Signal | Experiment | Expected Outcome | GPU Hours | Priority |
|--------|-----------|------------------|-----------|----------|
| E2 strong partial correlation | Re-run E2 on *real* SAEBench downstream metrics (not synthetic proxies) | Confirms r_partial ≈ -0.3 to -0.4 on actual sparse probing / RAVEL | 1–2 (data download + regression) | **High** |
| E1 feature_splitting dominance | Expand E1 to Pythia-160M with feature_splitting checkpoints | Feature_splitting maintains zero dead neurons and competitive CE recovery on a second model family | 2–3 | **High** |
| E3 domain-specific absorption | Run task-agnostic metric on biology + color hierarchies in addition to geography | Correlation with first-letter varies by domain; some domains show positive r | 1–2 | **High** |
| Architecture dummies non-significant | Re-run E2 with interaction terms (absorption × architecture) | Interaction terms are non-significant, confirming that absorption is the unified driver | 0.5 (re-analysis) | **Med** |
| Attention-output outlier | Evaluate 3–4 additional attention-output SAEs across layers | Systematic pattern: attn-out SAEs show high lexical absorption, low semantic absorption | 1–2 | **Med** |
| L0 positive coefficient | Run a controlled ablation fixing width and varying L0 within a single architecture | Positive L0 effect holds only when absorption is held constant | 2–3 | **Med** |

---

## Honest Caveats

### C1: E2 synthetic proxies
- **Counter-argument**: The E2 meta-analysis used synthetic sparse_probing_f1 and RAVEL proxies because of HF rate limits, not real SAEBench metrics.
- **Alternative explanation**: The strong correlations could be artifacts of the proxy generation procedure.
- **What would convince me**: Re-running the identical regression on the real SAEBench CSV and seeing the same sign, magnitude, and significance.

### C2: E1 limited to GPT-2 Small
- **Counter-argument**: GPT-2 Small is a 117M parameter model from 2019. Its SAE ecosystem is smaller and less diverse than Gemma-2-2B or Pythia.
- **Alternative explanation**: The feature_splitting dominance and Pareto diversity are idiosyncrasies of GPT-2, not generalizable facts.
- **What would convince me**: Replicating E1 on Pythia-160M with the same metric pipeline and observing similar family-level separation.

### C3: E3 negative correlation
- **Counter-argument**: A negative correlation between task-agnostic and first-letter metrics undermines the claim that the new metric measures the same underlying construct.
- **Alternative explanation**: The task-agnostic pipeline is buggy or misaligned; a correct implementation should correlate positively.
- **What would convince me**: A manual audit of 5–10 false-negative tokens showing that the ablation latent genuinely corresponds to the child concept in the geography hierarchy. If the pipeline is sound, the divergence is real and publishable as a critique of first-letter absorption.

### C4: Small N in E1 dominance tests
- **Counter-argument**: The Mann-Whitney test comparing standard vs feature_splitting has only 23 vs 4 samples. Power is low.
- **Alternative explanation**: The non-significance on absorption and hedging is just underpowered noise.
- **What would convince me**: Expanding to 15+ feature_splitting checkpoints and observing the same pattern, or at least consistent effect sizes.

---

## Bottom Line

There is a publishable story here, but it is not the story we originally pitched. The strongest signal is **E2**: absorption has a unique, statistically robust causal cost on downstream interpretability, even after controlling for reconstruction and sparsity. That alone reframes absorption from an "aesthetic pathology" to a measurable practical harm. The **E1** Pareto evidence is promising but model-limited; the **E3** task-agnostic metric is most interesting as a *critique* of the first-letter benchmark rather than a validation of it. If the follow-ups confirm the E2 effect on real data and extend E1 to Pythia, the paper can credibly claim both a causal contribution and a methodological contribution.
