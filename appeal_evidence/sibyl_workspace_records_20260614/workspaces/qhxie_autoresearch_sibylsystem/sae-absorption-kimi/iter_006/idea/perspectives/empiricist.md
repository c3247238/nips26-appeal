# Empiricist Perspective

## Phase 1: Literature Survey

### Methodology Resources

1. **[Chanin et al., 2024] "A is for Absorption" (arXiv:2409.14507, NeurIPS 2025 Oral)** --- Original absorption metric using first-letter probes with k-sparse probing (k=1..10), F1-threshold detection (tau_fs = 0.03), and absorption scoring via ground-truth probe projection. **Critical methodological limitation**: metric requires ground-truth labels, uses only first-letter tasks, and has never been validated on semantic hierarchies with controlled frequency.

2. **[Karvonen et al., 2025] SAEBench (arXiv:2503.09532)** --- Standardized 8-evaluation benchmark including absorption. **Methodological concern**: Absorption evaluation is computationally expensive (~26 min per SAE); uses first-letter tasks exclusively; supervised metrics examine "only a fraction of each SAE's latents"; authors explicitly refuse composite scoring due to "different scales and varying levels of noise."

3. **[Korznikov et al., 2026] "Sanity Checks for Sparse Autoencoders" (arXiv:2602.14111)** --- Introduced frozen-decoder, soft-frozen-decoder, and frozen-encoder random baselines. Found random baselines match trained SAEs on AutoInterp (0.87 vs 0.90), sparse probing (0.69 vs 0.72), and causal editing (0.73 vs 0.72). **Crucially: did NOT test absorption specifically.** On synthetic data, SAEs recover only 9% of true features despite 71% explained variance.

4. **[Wang et al., 2025] "Does Higher Interpretability Imply Better Utility?" (arXiv:2510.03659, ICLR 2026)** --- Pairwise analysis of 90 SAEs across 3 LLMs and 5 architectures. Found Kendall's tau_b ~ 0.298 between interpretability and steering utility; correlation collapses to ~0 (even negative) after feature selection. **Methodological strength**: largest-scale utility-interpretability correlation study to date.

5. **[Chanin & Garriga-Alonso, 2026] SynthSAEBench (arXiv:2602.14687)** --- Synthetic benchmark with 16,384 ground-truth features (10,884 hierarchical). Identified benchmark gaming: Matryoshka SAEs achieve best probing despite poor reconstruction; Matching Pursuit SAEs exploit superposition noise for better reconstruction without learning ground-truth features. **Methodological contribution**: first ground-truth hierarchy benchmark for SAE evaluation.

6. **[Chanin et al., 2025] "Feature Hedging" (arXiv:2505.11756)** --- Distinguished absorption (sparsity-driven, requires hierarchy, worse in wide SAEs) from hedging (reconstruction-driven, requires only correlation, worse in narrow SAEs). **Methodological insight**: hierarchy is an extreme form of correlation, but absorption requires strict containment while hedging requires only correlation.

7. **[Jedryszek & Crook, 2026] "Stable and Steerable Sparse Autoencoders" (arXiv:2603.04198)** --- Cross-seed consistency metrics (mean max cosine, fraction paired, shared features). Found SAE features vary drastically across seeds; L2 regularization improves consistency but causes latent collapse. **Methodological concern**: if features are not reproducible, any single-run absorption measurement is unreliable.

8. **[Kantamneni et al., 2025] "Are Sparse Autoencoders Useful?" (arXiv:2502.16681, ICML 2025)** --- SAEs won only 2.2% of 113 head-to-head sparse probing comparisons against simple baselines. **Methodological strength**: rigorous downstream utility evaluation with strong baselines.

9. **[Gulko et al., 2025] CE-Bench (arXiv:2509.00691)** --- Deterministic contrastive evaluation as alternative to LLM-based SAEBench metrics. Found >70% Spearman correlation with SAEBench but without LLM-evaluation noise. **Methodological contribution**: deterministic metrics avoid prompt instability and LLM-as-judge variance.

10. **[Leask et al., 2025] "Sparse Autoencoders Do Not Find Canonical Units of Analysis" (ICLR 2025)** --- SAE stitching and Meta-SAEs prove no single width learns a unique complete dictionary. **Methodological implication**: absorption measurements at one width may not generalize.

11. **[Cui et al., 2025] "On the Limits of Sparse Autoencoders" (arXiv:2506.15963)** --- Proved necessary and sufficient conditions for identifiability: extreme sparsity, sparse activation, sufficient hidden dimensions. Conditions are tight; counterexamples show identifiability fails when any condition is violated. **Methodological implication**: theoretical bounds on when absorption is unavoidable.

12. **[Chanin et al., 2025] "Sparse but Wrong" (arXiv:2508.16560)** --- Showed incorrect L0 estimation leads to incorrect features; at low L0, correct SAEs achieve worse reconstruction than incorrect ones. Sparsity-reconstruction tradeoff plots are misleading evaluation. **Methodological concern**: standard evaluation systematically biases toward incorrect feature learning.

### Experimental Landscape

**What has been properly tested:**
- First-letter absorption detection works reliably on models with "decent spelling knowledge" (>1B parameters) [Chanin et al., 2024; SAEBench docs]
- Matryoshka SAEs reduce first-letter absorption from ~0.49 to ~0.05 [Bussmann et al., 2025]
- OrtSAE reduces absorption by 65% with ~4-11% compute overhead [Korznikov et al., 2025]
- Random baselines match trained SAEs on AutoInterp, sparse probing, and RAVEL [Korznikov et al., 2026]
- Interpretability-utility correlation is weak (tau_b ~ 0.298) and collapses with feature selection [Wang et al., 2025]

**What is accepted without evidence:**
- That first-letter absorption generalizes to semantic hierarchies (no validation study exists)
- That absorption reduction predicts downstream utility (no systematic correlation study exists)
- That the absorption metric measures learned SAE structure rather than base-model geometry (no random-baseline decomposition exists for absorption specifically)
- That architectural innovations reduce absorption through the claimed mechanism (no component-isolated study exists)

**Where methodological gaps exist:**
1. **No random-baseline decomposition for absorption**: Korznikov et al. test AutoInterp/sparse probing/RAVEL but NOT absorption.
2. **No construct-validity study on semantic hierarchies**: The first-letter metric has never been validated on WordNet or other semantic hierarchies with controlled frequency.
3. **No causal component isolation**: Matryoshka combines multi-scale dictionaries, batch TopK, and hierarchical loss; no study isolates which component drives absorption reduction.
4. **No frequency-matched control**: Feature absorption is confounded by frequency differences between parent and child features; systematic frequency-matching studies are lacking.
5. **No reproducibility quantification**: Absorption measurements are single-run; cross-seed variance is unknown.

---

## Phase 2: Initial Candidates

### Candidate A: Component-Isolated Causal Analysis of Absorption Reduction (Front-Runner)

**Core hypothesis**: Among SAE architectural innovations claimed to reduce absorption, **multi-scale dictionary decomposition** (the core mechanism in Matryoshka SAEs) is the primary causal driver of absorption reduction, while **per-feature thresholding** (JumpReLU) and **gating mechanisms** (Gated SAE) have negligible independent effects (Cohen's d < 0.5 vs. baseline).

**Falsification criterion**: If a component other than multi-scale decomposition (e.g., orthogonality penalty, gating) shows the largest effect size (Cohen's d > 0.8) on absorption reduction, the hypothesis is falsified.

**Evaluation protocol**:
- Primary benchmark: SynthSAEBench-16k (ground-truth hierarchies: 128 root trees, depth 3, branching factor 4, 10,884 hierarchical features)
- Metrics: (1) Ground-truth absorption rate (fraction of parent features subsumed by child), (2) Feature recovery MCC, (3) Reconstruction MSE, (4) L0 sparsity
- Statistical test: One-way ANOVA across 6 variants (5 replicates each), post-hoc Tukey HSD, effect sizes (Cohen's d), Bonferroni correction across 15 pairwise comparisons
- Minimum random seeds: 5 per variant

**Ablation plan**:
| Variant | Components | What It Tests |
|---------|-----------|---------------|
| Baseline | Standard ReLU, L1 sparsity | Baseline absorption |
| +TopK | Replace L1 with TopK sparsity | Effect of explicit k-sparsity |
| +MultiScale | Nested dictionaries (2 levels) | Effect of hierarchical decomposition |
| +Orthogonality | Chunk-wise decoder orthogonality | Effect of decoder incoherence |
| +Gating | Decoupled detection/magnitude | Effect of gating mechanism |
| +Full Matryoshka | TopK + MultiScale + hierarchical loss | Combined effect (replicates prior) |

**Confounders identified**:
- Synthetic-to-real gap: SynthSAEBench features are constructed to be realistic but are still synthetic
- Component interactions: MultiScale + TopK may interact synergistically
- Training variance: Different random seeds may produce different feature dictionaries
- L0 mismatch: Different architectures may achieve different sparsity levels, confounding absorption comparisons

**Pilot design**: Baseline + TopK + MultiScale on SynthSAEBench-1k subset (1,024 features, 10M samples). Target: 15-20 min on 1 GPU. Hard gates: (1) All variants achieve MCC > 0.5, (2) Clear ordering: MultiScale < TopK < Baseline in absorption, (3) Variance across replicates < 20% of mean.

---

### Candidate B: Random-Baseline Decomposition of the SAEBench Absorption Metric

**Core hypothesis**: Random-decoder and PCA-basis SAEs will achieve absorption scores comparable to trained SAEs on semantic hierarchies (difference not significant at p > 0.05, Cohen's d < 0.5), indicating that much of the metric measures base-model geometry rather than learned structure.

**Falsification criterion**: If trained SAEs show significantly lower absorption than all random/PCA baselines (p < 0.05, Cohen's d > 0.5), the hypothesis is falsified.

**Evaluation protocol**:
- Primary benchmark: SAEBench absorption protocol on semantic hierarchies (WordNet hypernyms) and first-letter tasks
- Conditions: Trained SAE, Random-decoder SAE (frozen permuted decoder), PCA-basis SAE (decoder = top-k PCs), Fully random SAE (matched L0)
- Statistical test: One-way ANOVA across conditions, pairwise t-tests with Bonferroni correction
- Minimum SAEs: 6 architectures x 4 conditions = 24 measurements

**Ablation plan**:
- Ablation 1: Random-decoder only (tests geometric confound)
- Ablation 2: PCA-basis only (tests whether any structured basis works)
- Ablation 3: Fully random (tests base-model geometry alone)
- Ablation 4: Vary probe difficulty (AUROC 0.6-0.95 vs. 1.0) to test ceiling effects

**Confounders identified**:
- Probe difficulty ceiling effect: If all probes achieve AUROC = 1.0, absorption formula collapses to (1 - k_sparse_acc)
- Model dependence: GPT-2 vs. Pythia may show different patterns
- Hierarchy selection: Poorly chosen hierarchies may not trigger absorption even in valid metrics
- Permutation sensitivity: Random decoder initialization may vary across seeds

**Pilot design**: 3 conditions (trained, random-decoder, PCA) x 2 SAEs x 5 hierarchies. Target: 15-20 min. Hard gate: Random-decoder achieves <50% of trained SAE absorption OR p > 0.05 for trained vs. random difference.

---

### Candidate C: The Co-occurrence Confound --- Controlled Experiment Nobody Has Run

**Core hypothesis**: The absorption metric will show comparable or higher scores for high co-occurrence non-hierarchies than for true hierarchies when co-occurrence frequency is matched, demonstrating sensitivity to correlation rather than hierarchical containment.

**Falsification criterion**: If hierarchy absorption is significantly HIGHER than high co-occurrence non-hierarchy after frequency matching (p < 0.05, t > 0), the hypothesis is falsified.

**Evaluation protocol**:
- Primary benchmark: Custom semantic hierarchy dataset with frequency-matched controls
- Conditions: (1) True hierarchies (WordNet hypernyms), (2) High co-occurrence non-hierarchies (synonym pairs with matched parent-child frequency ratios), (3) Low co-occurrence non-hierarchies (random unrelated pairs), (4) Synthetic hierarchies with controlled vs. natural co-occurrence
- Statistical test: Paired t-test (hierarchy vs. high co-occurrence non-hierarchy), bootstrap 95% CI
- Minimum pairs: 10 per condition

**Ablation plan**:
- Ablation 1: Vary frequency ratio (parent/child frequency = 1.0, 2.0, 5.0, 10.0)
- Ablation 2: Vary hierarchy depth (2-level vs. 3-level)
- Ablation 3: Test on multiple SAE architectures to check consistency

**Confounders identified**:
- Frequency confound: Parent features are typically more frequent than child features; must match frequencies
- Probe quality: Poor probes may fail to detect true hierarchies
- Model-specific behavior: GPT-2 vs. Pythia may show different co-occurrence patterns
- Hierarchy ambiguity: Some "hierarchies" (e.g., animal -> pet) may be culturally variable

**Pilot design**: 6 SAEs x 3 conditions (hierarchy, high co-occurrence, low co-occurrence) x 5 pairs each. Target: 20-30 min. Hard gate: Non-hierarchy >= hierarchy (t <= 0) OR p > 0.05.

---

## Phase 3: Self-Critique

### Against Candidate A (Component-Isolated Analysis)

**Confound attack**: The synthetic-to-real gap is the biggest threat. SynthSAEBench features are designed to be realistic (Zipfian firing, correlation, superposition) but are still constructed. Real LLM features may have properties not captured by the synthetic generator. **Mitigation**: Phase 2 validation on real LLM; acknowledge limitation. Additionally, component interactions may dominate: MultiScale + TopK may work synergistically in ways neither does alone. The ANOVA design assumes additive effects.

**Statistical attack**: With 6 variants x 5 replicates = 30 data points, one-way ANOVA has adequate power (power ~0.80 for medium effect size, alpha = 0.05). However, 15 pairwise comparisons with Bonferroni correction require very large effect sizes to survive (effective alpha = 0.0033). **Mitigation**: Use Holm-Bonferroni (less conservative) or pre-register primary comparisons (MultiScale vs. Baseline, Full Matryoshka vs. Baseline). Report unadjusted p-values with explicit correction method.

**Benchmark attack**: SynthSAEBench is new (Feb 2026) and has limited community validation. The ground-truth hierarchy structure (128 trees, depth 3, branching factor 4) may not match real semantic hierarchies. **Mitigation**: Cite Chanin & Garriga-Alonso's validation against SAEBench; acknowledge as limitation; Phase 2 real-LLM validation provides external validity.

**Ablation completeness attack**: The ablation schedule tests 5 components but may miss interactions. For example, orthogonality + MultiScale may interact in ways neither does alone. The "Full Matryoshka" variant captures the combined effect but does not isolate all 2-way interactions. **Mitigation**: Add 2-3 interaction variants if pilot suggests interactions; otherwise, acknowledge limitation and frame as "main effects" study.

**Verdict**: STRONG. The ground-truth data eliminates probe-based metric collapse. The component-isolation design is novel. The synthetic-to-real gap is the main threat, but Phase 2 validation mitigates it.

---

### Against Candidate B (Random-Baseline Decomposition)

**Confound attack**: The prior iteration (iter_002) already attempted this and encountered fatal anomalies: Random=Standard identity, reversed H2, perfect AUROC ceiling effect. These anomalies suggest the metric pipeline on real LLMs is structurally flawed, not just the metric itself. **Mitigation**: The decomposition experiment should be run on models where the metric is known to work (>1B parameters per SAEBench docs). Alternatively, use SynthSAEBench where ground truth eliminates probe artifacts.

**Statistical attack**: If random baselines achieve comparable scores, the effect size will be small (Cohen's d < 0.5). With 6 SAEs x 4 conditions = 24 measurements, power to detect small effects is limited. **Mitigation**: Increase to 8-10 SAEs; use paired design (same SAE, different decoder conditions).

**Benchmark attack**: The SAEBench absorption metric was designed for first-letter tasks on >1B parameter models. Running it on semantic hierarchies or small models may produce invalid results by design. **Mitigation**: Test on both first-letter (where metric is validated) and semantic hierarchies (where generalization is claimed); compare patterns.

**Ablation completeness attack**: Random-decoder, PCA, and fully-random baselines test different aspects but may not isolate all confounds. For example, probe training quality, k-sparse probing threshold, and F1 detection threshold all contribute to the final score. **Mitigation**: Decompose the metric step-by-step (probe AUROC -> k-sparse accuracy -> absorption score) to identify which step introduces confounds.

**Verdict**: MODERATE. The prior iteration's anomalies suggest this experiment is risky on real LLMs. However, as a conceptual framework and a SynthSAEBench experiment (where probes are unnecessary), it remains valuable.

---

### Against Candidate C (Co-occurrence Confound)

**Confound attack**: Frequency matching is difficult. True hierarchies have natural frequency ratios (parent more frequent than child) that are hard to replicate in non-hierarchies. If frequency matching is imperfect, the experiment confounds hierarchy with frequency. **Mitigation**: Use synthetic data where frequencies can be precisely controlled; report frequency distributions for all conditions.

**Statistical attack**: With 10 pairs per condition and 6 SAEs, the paired t-test has 60 observations (10 pairs x 6 SAEs). Power is adequate for medium effects but may miss small differences. **Mitigation**: Use mixed-effects model (pair as random effect, SAE and condition as fixed effects) for better power.

**Benchmark attack**: The "absorption" measured on non-hierarchies is not theoretically defined. The metric was designed for hierarchies; applying it to non-hierarchies may produce meaningless numbers. **Mitigation**: Frame as a "control experiment" --- if the metric produces non-zero scores on non-hierarchies, this is evidence of confound, not a claim about "non-hierarchy absorption."

**Ablation completeness attack**: Co-occurrence is not the only possible confound. Semantic similarity, thematic relatedness, and distributional overlap may also trigger false positives. **Mitigation**: Include multiple control types (synonyms, antonyms, random pairs) to isolate co-occurrence specifically.

**Verdict**: STRONG. The prior iteration already found reversed H2 (non-hierarchy > hierarchy, t = -4.748, p = 0.0032), which is strong preliminary evidence. This candidate formalizes and extends that finding with proper controls.

---

## Phase 4: Refinement

**Dropped**: Candidate B (random-baseline decomposition on real LLMs) is demoted due to prior iteration anomalies suggesting the real-LLM pipeline is structurally flawed. The random-baseline insight is retained as a **control condition within Candidate A** (random-feature control on SynthSAEBench) rather than a standalone experiment.

**Strengthened survivors**:

1. **Candidate A (Component-Isolated Analysis)** is strengthened by:
   - Adding a random-feature control to the SynthSAEBench design (tests whether any structure is needed)
   - Pre-registering primary comparisons to avoid Bonferroni power loss
   - Adding L0-matching as a covariate to control for sparsity differences
   - Including reconstruction-sparsity-absorption Pareto analysis (not just absorption alone)

2. **Candidate C (Co-occurrence Confound)** is strengthened by:
   - Moving to SynthSAEBench where frequencies can be precisely controlled
   - Using mixed-effects models for better statistical power
   - Adding synthetic hierarchies with matched vs. unmatched frequency as a direct test

**Selected front-runner**: **Candidate A --- Component-Isolated Causal Analysis on SynthSAEBench-16k.**

Rationale: The empiricist prioritizes experiments where (a) ground truth is known, (b) confounds can be controlled, (c) causal claims are justified, and (d) results are reproducible. SynthSAEBench provides all four. The component-isolation design answers a question the community urgently needs answered: "What actually works?" The co-occurrence confound (Candidate C) is folded in as a **control experiment within the main study** --- testing whether absorption on SynthSAEBench hierarchies is specific to hierarchical containment or also triggered by correlated non-hierarchies.

---

## Phase 5: Final Proposal

### Title
**What Actually Reduces Feature Absorption? A Component-Isolated Study on Ground-Truth Synthetic Hierarchies**

### Hypothesis
Among SAE architectural innovations claimed to reduce absorption, **multi-scale dictionary decomposition** is the primary causal driver of absorption reduction (Cohen's d > 0.8 vs. baseline), while **per-feature thresholding** (JumpReLU) and **gating mechanisms** have negligible independent effects (Cohen's d < 0.5).

### Falsification Criterion
If any component other than multi-scale decomposition shows the largest effect size on absorption reduction (Cohen's d > 0.8), or if no component shows a significant effect (all p > 0.05 after correction), the hypothesis is falsified.

### Method

**Step 1: Baseline and Component-Isolated Variants**
Train 6 SAE variants on SynthSAEBench-16k, varying one component at a time:

| Variant | Components | What It Tests |
|---------|-----------|---------------|
| Baseline | Standard ReLU, L1 sparsity | Baseline absorption rate |
| +TopK | Replace L1 with TopK sparsity | Effect of explicit k-sparsity |
| +MultiScale | Nested dictionaries (2 levels) | Effect of hierarchical decomposition |
| +Orthogonality | Chunk-wise decoder orthogonality penalty | Effect of decoder incoherence |
| +Gating | Decoupled detection/magnitude paths | Effect of gating mechanism |
| +Full Matryoshka | TopK + MultiScale + hierarchical loss | Combined effect (replicates prior) |

**Step 2: Ground-Truth Absorption Measurement**
For each variant (5 replicates), measure:
- **Absorption rate**: Fraction of parent features subsumed by child features (using known ground-truth parent-child relationships)
- **Feature recovery MCC**: Matthews correlation between learned and ground-truth feature assignments
- **Reconstruction MSE**: Standard reconstruction quality
- **Sparsity (L0)**: Average active features per token

**Step 3: Control Experiments**
- **Random-feature control**: Train SAE with random decoder; expected MCC < 0.1 (validates metrics)
- **Co-occurrence control**: Measure "absorption" on correlated non-hierarchies vs. true hierarchies; if comparable, metric is confounded
- **L0-matched comparison**: Report absorption per unit L0 to control for sparsity differences

**Step 4: Statistical Analysis**
- One-way ANOVA across 6 variants (5 replicates each)
- Pre-registered primary comparisons: MultiScale vs. Baseline, Full Matryoshka vs. Baseline
- Post-hoc Tukey HSD for exploratory pairwise comparisons
- Effect sizes (Cohen's d) for each component vs. baseline
- Holm-Bonferroni correction across comparisons
- Mixed-effects model: variant (fixed) + replicate (random)

**Step 5: Real-LLM Validation (Phase 2)**
- Take top-performing component from Step 3
- Train on Pythia-160M or Gemma-2-2B with that component added to baseline
- Measure first-letter absorption via SAEBench
- Compare with existing architecture leaderboard

### Evaluation Protocol

**Primary benchmarks**: SynthSAEBench-16k (ground-truth hierarchies, 16,384 features, 10,884 hierarchical)

**Metrics with statistical test plan**:
| Metric | Test | Target |
|--------|------|--------|
| Absorption rate | One-way ANOVA + Tukey HSD | MultiScale < Baseline (p < 0.05) |
| Feature recovery MCC | One-way ANOVA + Tukey HSD | MultiScale > Baseline (p < 0.05) |
| Reconstruction MSE | One-way ANOVA + Tukey HSD | Full Matryoshka >= Baseline |
| L0 sparsity | Report as covariate | Matched across variants |
| Pareto frontier | Multi-objective plot | Best absorption-MCC-reconstruction trade-off |

**Number of random seeds**: Minimum 5 per variant (30 total for main experiment)

### Ablation Schedule

| Ablation | Component Removed | Expected Outcome |
|----------|-------------------|------------------|
| Baseline | None (Standard ReLU) | Highest absorption, lowest MCC |
| +TopK | L1 sparsity | Moderate absorption reduction |
| +MultiScale | Single-scale dictionary | Largest absorption reduction |
| +Orthogonality | Uncorrelated decoder | Small absorption reduction |
| +Gating | Coupled encoder | Negligible effect |
| +Full Matryoshka | All components combined | Lowest absorption, highest MCC |

### Control Experiments

1. **Random-feature control**: Validates that metrics discriminate structure from randomness
2. **Co-occurrence control**: Tests whether absorption is specific to hierarchies or triggered by any correlation
3. **L0-matched comparison**: Controls for sparsity differences across architectures
4. **Reconstruction-absorption Pareto**: Tests whether absorption reduction trades off against reconstruction

### Pilot Design

**Scope**: Baseline + TopK + MultiScale on SynthSAEBench-1k subset (1,024 features, 10M samples)
**Target duration**: 15-20 minutes on 1 GPU
**Hard gates**:
1. All variants achieve MCC > 0.5 (validates training)
2. Clear ordering: MultiScale < TopK < Baseline in absorption rate (validates hypothesis direction)
3. Variance across replicates < 20% of mean (validates reproducibility)
4. Random-feature control achieves MCC < 0.1 (validates metric discrimination)

### Resource Estimate

| Stage | Task | Duration | GPU-hours |
|-------|------|----------|-----------|
| Pilot | 3 variants x 1 replicate on 1k subset | 15-20 min | ~0.1 |
| Full | 6 variants x 5 replicates on 16k | ~60 min | ~0.5-1.0 |
| Validation | Top component on Pythia-160M | ~30 min | ~0.5 |
| Analysis | ANOVA + post-hoc + effect sizes | 5 min (CPU) | 0 |
| **Total** | | | **~1.0-1.5** |

All tasks well under the 1-hour limit per task.

### Risk Assessment

| Threat | Likelihood | Severity | Mitigation |
|--------|------------|----------|------------|
| Synthetic data doesn't match LLM feature structure | High | High | Phase 2 real-LLM validation; acknowledge in Discussion |
| Training instability (high variance across replicates) | Medium | Medium | 5 replicates; report variance; increase if needed |
| Component interactions dominate main effects | Medium | Medium | Test Full Matryoshka variant; report interactions if observed |
| No component shows significant effect | Low | High | Report null result as valuable; suggests absorption is not architecture-dependent |
| MultiScale trades absorption for hedging | Medium | Medium | Measure both absorption and hedging; report trade-off |
| Prior iteration data integrity issues recur | Medium | High | Pre-writing automated checks; no estimated data; all numbers from actual execution |

### Novelty Claim

This would be the **first component-isolated study of SAE architectural innovations using ground-truth synthetic hierarchies**. While Matryoshka SAEs, OrtSAE, and Gated SAEs have all reported absorption reductions, no study has isolated which specific component drives the improvement. By varying one component at a time on SynthSAEBench-16k --- where ground-truth features are known by construction --- we can make causal claims about what actually works. The synthetic-to-real validation tests whether these causal claims transfer to the community's primary use case.

**Specific empirical questions answered for the first time**:
1. Does multi-scale decomposition causally reduce absorption, or is it confounded with other Matryoshka components?
2. Do JumpReLU thresholding and Gating have independent effects on absorption?
3. Does orthogonality penalty reduce absorption on ground-truth hierarchies?
4. How do absorption, feature recovery, and reconstruction trade off across components?

---

## Sources

- [Chanin et al., 2024] "A is for Absorption" (arXiv:2409.14507)
- [Karvonen et al., 2025] SAEBench (arXiv:2503.09532)
- [Korznikov et al., 2026] "Sanity Checks for SAEs" (arXiv:2602.14111)
- [Wang et al., 2025] "Does Higher Interpretability Imply Better Utility?" (arXiv:2510.03659)
- [Chanin & Garriga-Alonso, 2026] SynthSAEBench (arXiv:2602.14687)
- [Chanin et al., 2025] "Feature Hedging" (arXiv:2505.11756)
- [Jedryszek & Crook, 2026] "Stable and Steerable SAEs" (arXiv:2603.04198)
- [Kantamneni et al., 2025] "Are Sparse Autoencoders Useful?" (arXiv:2502.16681)
- [Gulko et al., 2025] CE-Bench (arXiv:2509.00691)
- [Leask et al., 2025] "SAEs Do Not Find Canonical Units of Analysis" (ICLR 2025)
- [Cui et al., 2025] "On the Limits of SAEs" (arXiv:2506.15963)
- [Chanin et al., 2025] "Sparse but Wrong" (arXiv:2508.16560)
- [Bussmann et al., 2025] Matryoshka SAE (arXiv:2503.17547)
- [Korznikov et al., 2025] OrtSAE (arXiv:2509.22033)
