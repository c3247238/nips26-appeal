# Empiricist Perspective (Iteration 2)

## Phase 1: Literature Survey

### Methodology Resources

1. **[SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability](https://arxiv.org/abs/2503.09532)** (Karvonen et al., ICML 2025) — The most rigorous large-scale evaluation to date. Evaluates 200+ SAEs across 7 architectures with 8 metrics including feature absorption. Critical finding: sparsity-fidelity frontier does NOT predict downstream task performance. Rankings change dramatically by L0 regime and model scale.

2. **[A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders](https://arxiv.org/abs/2409.14507)** (Chanin et al., 2024/2025) — Foundational work defining feature absorption. Uses first-letter classification tasks with logistic regression probes as ground truth. Key methodological contribution: k-sparse probing with feature split detection (tau_fs = 0.03). Validated across hundreds of SAEs. Limitation: metric requires manual inspection; cannot measure past layer 17 in Gemma-2-2B due to information migration.

3. **[Sanity Checks for Sparse Autoencoders: Do SAEs Beat Random Baselines?](https://arxiv.org/abs/2602.14111)** (Korznikov et al., 2026) — Introduces three random baselines (frozen decoder, soft-frozen decoder, frozen encoder). Finding: frozen/random baselines achieve comparable performance to trained SAEs on Gemma-2-2B and Llama-3.1-8B across interpretability (0.87 vs 0.90), sparse probing (0.69 vs 0.72), and causal editing (0.73 vs 0.72). Synthetic test: only 9% true feature recovery despite 71% explained variance.

4. **[Does Higher Interpretability Imply Better Utility? A Pairwise Analysis on Sparse Autoencoders](https://arxiv.org/abs/2510.03659)** (Wang et al., ICLR 2026) — Measures correlation between interpretability and steering utility. Finding: Kendall's tau_b ~ 0.30 — weak positive correlation. Absorption metric reveals designs concentrating signal into few latents are less prone to feature stealing. Critical for any absorption study: must validate on downstream tasks, not just metrics.

5. **[Interpretability Illusions with Sparse Autoencoders](https://arxiv.org/abs/2505.16004)** (Li et al., 2025) — Shows SAE interpretations are vulnerable to minimal input perturbations via gradient-based attacks. LLM output remains stable while SAE activations change dramatically. Implication: absorption may be one manifestation of a broader reliability problem.

6. **[SynthSAEBench: Evaluating Sparse Autoencoders on Scalable Realistic Synthetic Data](https://arxiv.org/abs/2602.14687)** (Chanin et al., 2026) — Controlled synthetic benchmark with 16K ground-truth features. Enables ablations impossible with LLMs. Key finding: even in best-case scenarios (LRH holds by construction), no SAE architecture achieves perfect feature recovery. Discovers Matching Pursuit overfitting phenomenon.

7. **[Falsifying Sparse Autoencoder Reasoning Features in Language Models](https://arxiv.org/abs/2601.05679)** (2026) — Direct falsification study: 45-90% of "reasoning" features activate after injecting associated tokens into non-reasoning text. Features capture low-dimensional correlates, not genuine reasoning. Steering yields minimal benchmark changes.

8. **[CE-Bench: LLM-Free Contrastive Evaluation for Sparse Autoencoders](https://arxiv.org/abs/2509.00691)** (EMNLP-W 2025) — Addresses reproducibility crisis from LLM-dependent evaluation. Uses semantically contrastive story pairs for fully deterministic evaluation. 77.3% CRPR alignment with SAEBench.

9. **[Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders](https://arxiv.org/abs/2505.11756)** (Chanin et al., 2025) — Identifies feature hedging as distinct failure mode from absorption. Critical confounder: narrower SAEs reduce absorption but increase hedging, and vice versa. Any absorption study must control for hedging.

10. **[CorrSteer: Generation-Time LLM Steering via Correlated Sparse Autoencoder Features](https://arxiv.org/abs/2508.12535)** (2025) — Connects circuit discovery with steering. Uses correlation for feature selection, intervention for causality validation. Restricting selection to answer-generation activations reduces spurious correlations.

11. **[On the Limits of Sparse Autoencoders: A Theoretical Framework and Reweighted Remedy](https://arxiv.org/abs/2506.15963)** (Cui et al., 2025) — Formal identifiability analysis. Proves conditions for recovering ground-truth features. Theoretical; limited empirical validation on LLM SAEs.

12. **[Evaluating SAE Interpretability Without Explanations](https://arxiv.org/abs/2507.08473)** (2025) — Explanation-free interpretability evaluation. Addresses reliability concerns of LLM-generated explanations.

### Experimental Landscape

**What has been properly tested:**
- Feature absorption EXISTS and is widespread (Chanin et al., SAEBench)
- Absorption increases with SAE width and sparsity (Chanin et al., SAEBench)
- JumpReLU and TopK worsen absorption compared to ReLU (SAEBench)
- Matryoshka SAEs reduce absorption from 0.49 to 0.05 (Bussmann et al.)
- OrtSAE reduces absorption by 65% (Korznikov et al.)
- Random baselines match trained SAEs on key metrics (Korznikov et al., 2026)

**What is accepted without proper evidence:**
- That absorption rates vary systematically across model layers (Chanin found no clear pattern, but limited to layers 0-17)
- That reducing absorption improves downstream interpretability tasks (Wang et al. show weak correlation ~0.3)
- That absorption findings on first-letter tasks generalize to real semantic features
- That synthetic benchmark absorption rates correspond to real LLM rates
- That current absorption metrics distinguish real learning from dictionary-size artifacts

**Where methodological gaps exist:**
- No systematic cross-model, cross-layer quantification of absorption prevalence (Gap 1 in literature survey)
- No work has systematically measured how absorption affects circuit discovery, concept erasure, or steering (Gap 2)
- No validation that absorption metrics beat random baselines (Gap 7 — Sanity Checks challenge)
- No longitudinal/temporal studies of absorption in pretrained SAEs (Gap 6)
- No cross-architecture comparison using identical evaluation protocols (Gap 10)

---

## Phase 2: Initial Candidates

### Candidate A: Systematic Absorption Prevalence Quantification Across Model Scales and Layers

- **Hypothesis**: Feature absorption rates vary systematically with model scale, layer depth, and SAE configuration in a predictable pattern that can be characterized and used as reference baselines for the community.
- **Falsification criterion**: If absorption rates show no consistent pattern across models/layers (random variation within measurement noise), or if random/frozen SAE baselines show comparable "absorption" patterns, the hypothesis is disproven.
- **Evaluation protocol**:
  - Primary: SAEBench absorption metric on GemmaScope SAEs (Gemma-2-2B, 9B) across all measurable layers
  - Secondary: OpenAI GPT-2 SAEs, LlamaScope SAEs for cross-model validation
  - Statistical: Bootstrap 95% CIs on absorption rates, paired comparison across layer depths
  - Random baseline control: Run identical absorption metric on frozen-decoder SAE baselines (per Korznikov et al.)
- **Ablation plan**:
  - Ablate by SAE architecture (ReLU, TopK, JumpReLU) — tests whether absorption is architecture-specific
  - Ablate by dictionary size (16K, 65K, 131K, 1M) — tests width dependence
  - Ablate by sparsity level (L0 regimes: low <100, mid 100-500, high >500) — tests sparsity dependence
  - Ablate by model scale (2B vs 9B) — tests scale dependence
- **Confounders**:
  - Feature hedging (narrow SAEs show hedging instead of absorption — must control for width)
  - Information migration in later layers (first-letter task fails past layer 17 — limits layer coverage)
  - Dictionary-size artifacts (random baselines may show spurious absorption-like patterns)
  - Training data differences across SAE suites
- **Pilot design**: Run SAEBench absorption metric on one Gemma-2-2B layer (layer 12) with 16K and 65K JumpReLU SAEs. Compare with frozen-decoder baseline. Expected time: ~15 min per SAE (26 min metric + 33 min setup, but setup amortized).

### Candidate B: Controlled Experiment on Absorption's Impact on Downstream Circuit Discovery

- **Hypothesis**: Higher absorption rates in SAEs directly degrade the accuracy of circuit discovery and steering interventions, and this degradation is measurable and statistically significant.
- **Falsification criterion**: If circuit discovery accuracy shows no correlation with absorption rates across SAE configurations, or if random/frozen SAEs achieve comparable circuit discovery accuracy, the hypothesis is disproven.
- **Evaluation protocol**:
  - Primary: Circuit discovery accuracy using attribution patching on known circuits (e.g., indirect object identification, greater-than comparison) with SAE feature representations vs. raw residual stream
  - Secondary: Steering fidelity on targeted behaviors (sentiment, factual recall) — measure success rate vs. absorption rate
  - Statistical: Pearson/Spearman correlation between absorption rate and circuit/steering accuracy; bootstrap CIs
  - Control: Compare against random/frozen SAE baseline; compare against non-SAE probing baseline
- **Ablation plan**:
  - Ablate by absorption level (select SAEs with low/medium/high absorption via SAEBench) — tests dose-response
  - Ablate by circuit complexity (simple vs. multi-hop) — tests whether absorption affects complex circuits more
  - Ablate by intervention type (ablation vs. steering vs. replacement) — tests generality
- **Confounders**:
  - Reconstruction quality covaries with absorption (better reconstruction often means more absorption) — must partial out
  - Circuit discovery method itself may be flawed — use established, validated circuits only
  - Steering success depends on feature selection method — hold selection method constant
  - Spurious correlation: SAEs with high absorption may also have other flaws
- **Pilot design**: Test one known circuit (indirect object identification) on Gemma-2-2B layer 12 with two SAEs: one low-absorption (Matryoshka) and one high-absorption (standard JumpReLU). Measure circuit edge recovery accuracy. Expected time: ~10-15 min.

### Candidate C: The "Sanity Check" Challenge — Do Absorption Metrics Beat Random Baselines?

- **Hypothesis**: Current feature absorption metrics (Chanin et al., SAEBench) genuinely distinguish trained SAEs from random/frozen baselines, indicating they measure a real phenomenon rather than dictionary-size artifacts.
- **Falsification criterion**: If frozen-decoder or soft-frozen-decoder SAEs show absorption rates statistically indistinguishable from trained SAEs on the same metric, the hypothesis is disproven and absorption metrics are artifacts.
- **Evaluation protocol**:
  - Primary: Run identical absorption metric on (a) trained GemmaScope SAEs, (b) frozen-decoder baselines, (c) soft-frozen-decoder baselines
  - Secondary: Extend to multiple architectures (ReLU, TopK, JumpReLU) and widths (16K, 65K)
  - Statistical: Two-sample t-test or Mann-Whitney U test comparing trained vs. random baseline absorption rates; effect size (Cohen's d)
  - Multiple random seeds for baseline initialization (minimum 3)
- **Ablation plan**:
  - Ablate by initialization scheme (iso vs. cov per Korznikov) — tests sensitivity
  - Ablate by decoder constraint strength (frozen vs. soft-frozen with varying cosine thresholds) — tests gradient
  - Ablate by metric variant (mean absorption vs. full absorption rate vs. partial absorption) — tests which metric is most discriminative
- **Confounders**:
  - Frozen decoder baselines still learn encoder patterns that may mimic absorption structure
  - Different initialization schemes may produce different baseline behavior
  - The absorption metric itself may be biased toward finding "absorption" in any overcomplete dictionary
- **Pilot design**: Train one frozen-decoder JumpReLU SAE on Gemma-2-2B layer 12 (16K width) and compare absorption rate to trained GemmaScope equivalent. Expected time: ~15 min training + 26 min evaluation.

---

## Phase 3: Self-Critique

### Against Candidate A (Systematic Prevalence Quantification)

- **Confound attack**: The first-letter classification task used for absorption measurement may not generalize to real semantic features. It tests a very specific hierarchical structure (letter -> word) that may not represent the diversity of feature hierarchies in LLMs. Additionally, Chanin et al. already found no clear layer pattern — this candidate risks rediscovering their null result with more compute.
- **Statistical attack**: With absorption rates potentially varying widely across SAEs, detecting a systematic pattern requires large sample sizes per condition. The planned bootstrap CIs may be wide unless many SAEs are tested. Power analysis needed: how many SAEs per (model, layer, width, architecture) cell?
- **Benchmark attack**: SAEBench's absorption metric is explicitly noted as "diagnostic" rather than practically relevant. Measuring it at scale produces a large diagnostic dataset, but the community may question its value. The metric also cannot measure past layer 17 in Gemma-2-2B, creating a systematic bias in layer coverage.
- **Ablation completeness attack**: The ablation plan covers architecture, width, sparsity, and model scale — but misses training data size, training duration, and feature frequency distribution. GemmaScope SAEs were trained with different hyperparameters across scales, confounding scale with training differences.
- **Verdict**: MODERATE — valuable baseline data but risks being a "measurement paper" without clear theoretical or practical advance. Must include random baseline comparison to address Sanity Checks challenge.

### Against Candidate B (Downstream Impact on Circuit Discovery)

- **Confound attack**: Circuit discovery accuracy depends on many factors beyond absorption: reconstruction quality, feature density, sparsity level, feature interpretability. Absorption is just one variable in a multivariate system. Without controlling for reconstruction quality and sparsity, any correlation with circuit accuracy may be spurious.
- **Statistical attack**: The expected effect size is unclear. Wang et al. found tau_b ~ 0.3 for interpretability-utility correlation. Absorption-circuit accuracy correlation may be even weaker. With limited SAE configurations available (maybe 5-10 distinct absorption levels), detecting weak correlations requires careful design.
- **Benchmark attack**: Known circuits (IOI, greater-than) are themselves debated — some argue they are post-hoc rationalizations. Using them as ground truth assumes the circuit literature is correct, which is itself contested. Additionally, circuit discovery on SAE features is a relatively new method with its own validity questions.
- **Ablation completeness attack**: The dose-response design (low/medium/high absorption SAEs) is good, but "absorption level" is not independently manipulable — it covaries with architecture and training. A true causal test would require manipulating absorption while holding other factors constant, which is impossible without retraining SAEs (violating training-free constraint).
- **Verdict**: WEAK — the causal inference problem is severe. Cannot manipulate absorption independently. Correlation without manipulation cannot establish causation. Would need to pivot to a different design.

### Against Candidate C (Sanity Check Challenge)

- **Confound attack**: Frozen decoder baselines still train the encoder, which may learn to mimic absorption-like patterns. The soft-frozen decoder (cosine similarity >= 0.8 constraint) is closer to random but still allows some learning. A truly random baseline would freeze both encoder and decoder, which would produce trivial results (no reconstruction). The comparison space is narrow.
- **Statistical attack**: The key question is effect size. If trained SAEs show absorption rate 0.30 and frozen baselines show 0.25, is that meaningful? The difference may be statistically significant with enough samples but practically negligible. Need to define a meaningful effect size threshold a priori.
- **Benchmark attack**: The absorption metric uses first-letter tasks, which may be particularly susceptible to dictionary-size artifacts. Large random dictionaries will inevitably contain vectors that correlate with any probe direction by chance. The metric may be measuring "dictionary size" more than "absorption."
- **Ablation completeness attack**: Testing multiple initialization schemes and constraint strengths is good, but misses the question of whether ANY overcomplete sparse dictionary shows absorption-like patterns. A purely random Gaussian dictionary (no training at all) should also be tested as an extreme baseline.
- **Verdict**: STRONG — addresses the most pressing methodological challenge in the field. Even a null result (no difference between trained and random) would be highly publishable and important. The design is clean, falsifiable, and directly responds to a 2026 paper. However, must include the pure random dictionary extreme baseline.

---

## Phase 4: Refinement

### Dropped

- **Candidate B (Downstream Impact)** is dropped due to the fundamental causal inference problem. Without the ability to manipulate absorption independently (requires retraining SAEs, violating the training-free constraint), any correlation between absorption and circuit accuracy is hopelessly confounded. The field already has Wang et al.'s weak correlation result; adding another correlational study adds little.

### Strengthened

- **Candidate A** is strengthened by incorporating the random baseline comparison from Candidate C. The systematic quantification MUST include frozen/soft-frozen decoder baselines at each (model, layer, width) condition. This transforms it from a pure measurement study into a study that also validates whether the measurement itself is meaningful.

- **Candidate C** is strengthened by:
  1. Adding a pure random dictionary baseline (no training at all) — Gaussian random decoder matrix, test absorption metric directly
  2. Testing multiple metric variants (mean absorption, full absorption rate, partial absorption) to see which are most discriminative
  3. Including a "dictionary size sweep" — test whether absorption rates scale with dictionary size in trained vs. random baselines differently
  4. Connecting to synthetic validation: test on SynthSAEBench where ground-truth features are known, compare absorption metric behavior on true vs. random features

### Selected Front-Runner: Hybrid of A + C

**Title**: "Do Feature Absorption Metrics Measure Real Phenomena? A Systematic Validation Against Random Baselines Across Model Scales and Architectures"

This combines the systematic quantification of Candidate A with the rigorous validation of Candidate C. The core contribution is NOT just measuring absorption rates, but validating that the measurement tool itself works — that it distinguishes trained SAEs from random baselines in a way that correlates with known ground truth.

---

## Phase 5: Final Proposal

### Title
**"Validating Feature Absorption Metrics: A Systematic Cross-Scale Study with Random Baseline Controls"**

### Hypothesis
Feature absorption metrics (Chanin et al., SAEBench) produce significantly lower absorption rates for trained SAEs compared to random/frozen decoder baselines of identical size, and this difference correlates with ground-truth feature recovery on synthetic benchmarks. Formally: for trained SAEs vs. frozen-decoder baselines at matched (model, layer, width, architecture), the mean absorption rate difference is > 0.05 with Cohen's d > 0.5.

### Falsification Criterion
If trained SAEs and frozen-decoder/soft-frozen-decoder baselines show statistically indistinguishable absorption rates (p > 0.05, |Cohen's d| < 0.2) across all tested conditions, the hypothesis is falsified. This would imply absorption metrics are dictionary-size artifacts, not measures of real SAE behavior.

### Method
Training-free analysis of pretrained SAEs (GemmaScope, OpenAI GPT-2, LlamaScope) with three types of baseline comparisons:

1. **Frozen decoder baseline**: Random decoder initialization, frozen during training (encoder trained)
2. **Soft-frozen decoder baseline**: Random decoder initialization, constrained to cosine similarity >= 0.8 with initial value
3. **Pure random baseline**: Untrained Gaussian random decoder matrix (no training at all)

For each baseline type, generate 3 random seeds. Run identical SAEBench absorption metric protocol on trained and baseline SAEs.

### Evaluation Protocol

**Primary Benchmarks**:
- SAEBench feature absorption metric on GemmaScope SAEs (Gemma-2-2B, layers 5, 12, 19; widths 16K, 65K)
- OpenAI GPT-2 small SAEs (layers 6, 8, 10; multiple widths)
- SynthSAEBench-16K synthetic validation (ground-truth features known by construction)

**Metrics with Statistical Test Plan**:
- Mean absorption rate (1 - absorption score, higher is better)
- Full absorption rate (fraction of latents with any absorption)
- Per-letter absorption breakdown
- Statistical tests: Two-sample t-test or Mann-Whitney U (depending on normality); Cohen's d for effect size; bootstrap 95% CIs (1000 resamples)
- Minimum 3 random seeds per baseline type

**Control Experiments**:
- Dictionary size sweep: Test absorption rates at 4K, 16K, 65K, 131K latents for both trained and random baselines
- Synthetic ground-truth validation: On SynthSAEBench, compare absorption metric behavior when ground-truth features are known vs. random
- Feature hedging control: Measure hedging rates alongside absorption to ensure the two failure modes are distinguishable

### Ablation Schedule

| Component | What It Tests | Expected Outcome |
|-----------|--------------|------------------|
| Architecture (ReLU vs TopK vs JumpReLU) | Whether absorption metric discrimination is architecture-dependent | JumpReLU/TopK may show smaller trained-random gaps due to higher baseline absorption |
| Width (16K vs 65K vs 131K) | Whether dictionary-size artifacts dominate at scale | Larger widths may show smaller effect sizes (random baselines improve) |
| Layer depth (early vs middle vs late) | Whether information migration affects metric validity | Later layers may show noisier results due to first-letter task limitations |
| Model scale (2B vs 9B) | Whether scale affects absorption metric behavior | 9B may show clearer patterns due to more structured features |
| Baseline type (frozen vs soft-frozen vs pure random) | Gradient of "randomness" vs. absorption discrimination | Pure random should show highest absorption; trained lowest |

### Pilot Design
Run SAEBench absorption metric on:
1. One trained Gemma-2-2B layer 12 JumpReLU SAE (16K width)
2. One frozen-decoder baseline (same config, seed 0)
3. One pure random baseline (same config)

Compare absorption rates. Expected time: ~15 min per SAE (amortized setup). Total pilot: ~45 min.

### Resource Estimate
- GPU: NVIDIA RTX 3090 or equivalent (local GPU available)
- Per SAE evaluation: ~26 min (absorption metric) + ~33 min setup (amortized across batch)
- Total SAEs: ~30 (trained: 12; frozen baselines: 9; soft-frozen: 9)
- Total GPU time: ~15-20 hours (can be batched and run overnight)
- Within project constraints: single experiments < 1 hour; pilot < 15 min

### Risk Assessment

| Threat | Likelihood | Mitigation |
|--------|-----------|------------|
| Random baselines match trained SAEs on absorption | Medium | This IS the research question; a null result is still publishable and important |
| First-letter task fails in late layers | High | Limit analysis to layers 0-17 where metric is valid; report this limitation explicitly |
| SAEBench setup overhead dominates | Medium | Batch evaluations to amortize setup; use SAEBench's single-script automation |
| Frozen decoder training is unstable | Low | Use Korznikov et al.'s published code and hyperparameters; test multiple seeds |
| Synthetic-real gap invalidates SynthSAEBench validation | Medium | Treat SynthSAEBench as supplementary; primary claims based on real LLM SAEs |
| Feature hedging confounds absorption measurement | Medium | Measure both; report hedging rates; exclude narrow SAEs where hedging dominates |

### Novelty Claim
This study would be the first to systematically validate whether feature absorption metrics distinguish trained SAEs from random baselines across multiple model scales, architectures, and dictionary sizes. It directly responds to the "Sanity Checks" challenge (Korznikov et al., 2026) and provides either (a) evidence that absorption is a real, measurable phenomenon, or (b) evidence that current absorption metrics are artifacts of overcomplete dictionary sizes. Either outcome advances the field by resolving a critical methodological uncertainty.

### Additional Controls (Post-Refinement)

1. **Reconstruction quality matching**: Ensure trained and baseline SAEs are compared at matched reconstruction quality (explained variance), not just matched size. If random baselines have much worse reconstruction, lower absorption could be a trivial consequence.

2. **Sparsity level matching**: Compare at matched L0 sparsity, not just matched width. Different architectures achieve different L0 at the same width.

3. **Feature frequency stratification**: Test absorption separately for high-frequency vs. low-frequency features. Absorption may primarily affect low-frequency features; random baselines may not show this pattern.

4. **Cross-metric validation**: Compare absorption metric results with independent measures of feature quality (sparse probing F1, AutoInterp scores) on the same SAEs. If absorption and these metrics agree on ranking trained vs. random, it strengthens validity.

### Connection to Project Constraints

- **Training-free**: All analysis uses existing pretrained SAEs or generates random/frozen baselines without SAE retraining
- **Model scale**: GPT-2, Gemma-2B (primary), Gemma-9B (secondary if time permits)
- **Time budget**: Single SAE evaluation ~26 min; pilot ~45 min; full study batched overnight
- **Toolchain**: SAELens (loading), SAEBench (evaluation), TransformerLens (activation extraction), custom baseline generation
