# Pragmatist Perspective

## Phase 1: Literature Survey

### Key Resources Found

1. **[tml-epfl/why-weight-decay](https://github.com/tml-epfl/why-weight-decay)** (NeurIPS 2024, MIT license) -- The best-maintained open-source codebase for WD experiments. Includes ResNet-20/VGG-16-BN on CIFAR-10/100 and ResNet-50 on ImageNet, with full weight/gradient norm tracking infrastructure. **Has usable code.** This is our starting point for the evaluation framework.

2. **[zeke-xie/stable-weight-decay-regularization](https://github.com/zeke-xie/stable-weight-decay-regularization)** (NeurIPS 2023, MIT license) -- The AdamS/SWD optimizer implementation. CIFAR-10/100 experiments with gradient-norm-aware dynamic WD. **Has usable code.** Direct WD scheduling baseline.

3. **[CWD (Cautious Weight Decay)](https://arxiv.org/abs/2510.12402)** (ICLR 2026) -- One-line modification to AdamW. Available in `pytorch_optimizer` package by kozistr and various community repos. **Has usable code** (one-line drop-in). The strongest alignment-aware baseline.

4. **[hed-ucas/AlphaDecay](https://github.com/hed-ucas/AlphaDecay)** (Apache-2.0) -- Module-wise adaptive WD for LLMs using spectral density analysis (HT-SR theory). **Has usable code.** Relevant for norm-matched WD sub-approach characterization, though LLM-focused.

5. **[AlbertoFdezHdez/OUI](https://github.com/AlbertoFdezHdez/OUI)** (MIT license) -- Overfitting-Underfitting Indicator for WD quality monitoring during training. DenseNet/EfficientNet/ResNet experiments. **Has usable code.** Useful diagnostic component.

6. **[Weight Norm Control / AdamWN](https://arxiv.org/abs/2311.11446)** (Loshchilov 2023) -- Generalizes decoupled WD to target-norm control. Key theoretical framework for norm-matched WD. No standalone repo but described precisely enough to implement in ~20 lines of PyTorch.

7. **[Rotational Equilibrium](https://arxiv.org/abs/2305.17212)** (Kosson et al. 2023) -- WD induces balanced rotation across layers/neurons. Critical for understanding *why* WD works and for defining the Coupling Stability Index. Theory paper, no standalone implementation needed.

8. **[Correction of Decoupled Weight Decay](https://arxiv.org/abs/2512.08217)** (Chou 2025) -- Derives gamma^2 WD scaling for stable weight norm. Total Update Contribution analysis. Practical scaling rule for norm-matched WD.

9. **[FAdam](https://arxiv.org/abs/2405.12807)** (Hwang 2024) -- Shows Adam's v_t approximates diagonal Fisher. Derives corrections to WD based on Fisher geometry. Submitted to ICLR 2025 but withdrawn. The core insight (v_t as Fisher proxy for WD reweighting) is sound but the practical gains were apparently not convincing enough for reviewers.

10. **[Fishers for Free?](https://arxiv.org/abs/2507.18807)** (2025) -- Rigorous assessment of whether Adam's squared gradient accumulator is a valid Fisher diagonal proxy. Empirically validates across 6 settings. **Critical for assessing the viability of Fisher-based WD reweighting.**

11. **[WeightWatcher](https://weightwatcher.ai/)** -- Open-source data-free DNN diagnostic tool. Spectral analysis of weight matrices. Could complement our visualization framework but focuses on post-training analysis rather than during-training monitoring.

12. **[Fantastic Pretraining Optimizers and Where to Find Them](https://arxiv.org/abs/2509.02046)** (Wen et al. 2025) -- Systematic comparison of 10 optimizers across 0.1B-1.2B scales. Key finding: speedups of novel optimizers over well-tuned AdamW shrink from 1.4x at 0.1B to 1.1x at 1.2B. **Critical reality check for any optimizer modification claim.**

### Landscape Summary

Here is what actually works vs. what sounds good on paper:

**What works reliably:**
- AdamW with decoupled WD is the undisputed default. Every benchmark comparison starts here.
- CWD (one-line sign-alignment mask) provides consistent 0.1-0.6% accuracy improvements and lower final loss across scales. This is the current SOTA for simple WD modifications.
- Well-tuned WD values (1e-4 for SGD on CIFAR, 0.01-0.1 for AdamW on transformers) remain remarkably hard to beat with dynamic schemes.

**What sounds promising but has caveats:**
- SWD/AdamS improves over baseline on CIFAR but the improvement is "marginal for complex loss landscapes such as ImageNet" (the authors' own admission).
- Fisher-based WD reweighting (FAdam's approach) is theoretically motivated but the paper was *withdrawn* from ICLR 2025 -- the practical gains did not survive rigorous review.
- Optimizer speedups shrink dramatically with scale (Wen et al. 2025) -- a 1.4x speedup at 0.1B becomes 1.1x at 1.2B. Any WD modification claiming large improvements should be viewed with extreme suspicion at scale.

**What is actually broken/missing:**
- There is genuinely no standardized way to compare WD methods across papers. Every paper picks different architectures, datasets, training budgets, and metrics.
- Nobody has done a comprehensive head-to-head comparison of all major WD variants on the same codebase with identical compute budgets.
- The visualization/diagnostic side is genuinely underdeveloped. Weight norm trajectories exist in individual papers but no systematic toolkit exists.

**Engineering reality check:**
- The `why-weight-decay` repo (NeurIPS 2024) is the best starting framework -- it already tracks weight norms, gradient norms, and has CIFAR/ImageNet infrastructure.
- CWD is literally one line of code. SWD/AdamS is an available optimizer. AdamWN is ~20 lines. These are all implementable in a day.
- The hard part is not implementing the methods; it is running controlled experiments with identical compute budgets and producing publication-quality visualizations.


## Phase 2: Initial Candidates

### Candidate A: Comprehensive Benchmark + Standardized Metrics (The "Strong Baseline Done Right" Paper)

- **Core hypothesis**: A fair, compute-controlled comparison of all major dynamic WD methods on a shared codebase will reveal that (1) the performance differences are smaller than individual papers claim, (2) a simple standardized metric suite (BEM, CSI, AIS) explains the rankings better than final accuracy alone, and (3) the most impactful contribution is the diagnostic framework itself.

- **Implementation sketch**: Fork `why-weight-decay` repo. Implement all baselines (AdamW, CWD, SWD/AdamS, AdamWN, AlphaDecay-style per-layer WD, cosine WD schedule) on top of the same training infrastructure. Add metric computation: weight norm trajectories, gradient-weight cosine similarity, effective learning rate, spectral condition number. Define and compute BEM, CSI, AIS. Run all methods on the same compute budget per experiment.

- **Simplest version**: Compare just AdamW, CWD, and SWD on CIFAR-10/ResNet-20 with 3 seeds, tracking weight norm + alignment cosine + gradient norm. If the metrics already differentiate the methods meaningfully, the approach is validated.

- **Time estimate**: Pilot on CIFAR-10 (6 methods x 3 seeds x 15 min) = 4.5 GPU-hours. Full suite on CIFAR-10/100 with 2 architectures = ~18 GPU-hours. ImageNet = ~72 GPU-hours. Total ~95 GPU-hours, 2-3 days on 8 GPUs.

- **Reusable components**: `why-weight-decay` repo (training infra), CWD (one-line), SWD (AdamS optimizer), PyTorch built-in AdamW/SGD, standard CIFAR/ImageNet loaders.

### Candidate B: Unified Mathematical Framework with Practical Instantiations

- **Core hypothesis**: All four WD sub-approaches (scheduling, alignment-aware, decoupled, norm-matched) can be expressed as special cases of a general update rule `lambda_eff(i,t) = f(w_i, g_i, t, tau)`, and this unification reveals a natural "best combination" that cherry-picks the load-bearing component from each sub-approach.

- **Implementation sketch**: Define the general WD update rule. Derive each existing method as a special case. Propose a combined method that uses: (a) cosine scheduling from the scheduling literature, (b) CWD's sign-alignment mask, (c) decoupled application a la AdamW, (d) per-layer target norm from norm-matched WD. Test whether the combination outperforms any individual component. The combined method requires ~30 lines of code.

- **Simplest version**: Implement CWD + cosine WD schedule (two modifications that are independently validated). Test whether the combination is additive (both help independently) or redundant (alignment mask already captures what scheduling does).

- **Time estimate**: Same compute as Candidate A since the methods need comparison anyway. The theoretical derivation is the main additional work (1-2 days of math).

- **Reusable components**: Same as Candidate A, plus LaTeX for mathematical derivations.

### Candidate C: Fisher-Reweighted Weight Decay (Practical TWD)

- **Core hypothesis**: Using Adam's second-moment estimate v_t to reweight per-parameter WD strength (`lambda_eff(i) = lambda_0 * sqrt(v_t(i)) / RMS(sqrt(v_t))`) will improve over standard AdamW by applying stronger decay to high-curvature parameters and weaker decay to low-curvature parameters, achieving better thermodynamic efficiency of regularization.

- **Implementation sketch**: Modify AdamW's weight decay step to use the formula above. One-line change in the optimizer. Test on CIFAR-10/100 and ImageNet. Compare against standard AdamW and CWD.

- **Simplest version**: Implement the one-line Fisher-reweighted WD in AdamW. Run CIFAR-10/ResNet-20 with 3 seeds. If it does not improve over standard AdamW by at least 0.1% accuracy, the approach is dead.

- **Time estimate**: Implementation: 30 minutes. Testing: same as Candidate A pilot (4.5 GPU-hours for basic validation).

- **Reusable components**: Standard PyTorch AdamW (modify 1 line), same training infrastructure.


## Phase 3: Self-Critique

### Against Candidate A (Comprehensive Benchmark)

- **Implementation reality check**: The `why-weight-decay` repo exists and is well-maintained. CWD is one line. SWD has a repo. This is entirely implementable. The main engineering challenge is ensuring truly fair comparison -- same data augmentation, same LR schedule, same compute budget per method. This requires careful attention to detail but no novel engineering.

- **Reproducibility attack**: This is the *most* reproducible candidate. All baselines are open-source. The metric definitions are mathematical. Anyone with the same GPUs can reproduce. The only risk is that different WD methods may prefer different LR values, and a "fair" comparison requires either (a) tuning LR independently per method (expensive) or (b) using the same LR for all (potentially unfair to some methods).

- **Baseline sanity check**: The risk is that the paper's conclusion is "all methods perform about the same when properly tuned." This is actually a valuable finding (the field *needs* someone to say this), but it may be hard to publish as a top venue paper without a stronger theoretical contribution. SWD's own authors admit marginal gains on ImageNet. CWD gains are 0.1-0.6%. If we show these differences are within noise with proper compute-controlled comparison, that is a legitimate negative result -- but negative results are hard to publish.

- **Scope attack**: CIFAR + ImageNet + multiple architectures gives broad scope. The benchmark paper format (Benchmarking X: A Comprehensive Evaluation) has well-established precedent in the optimization literature. Adding the standardized metrics (BEM, CSI, AIS) differentiates this from a simple comparison.

- **Verdict**: STRONG -- This is the highest-probability-of-success candidate. The infrastructure exists, the experimental protocol is standard, and the contribution (standardized comparison + metrics) is genuinely needed. The risk is that reviewers want a novel method, not just a benchmark.

### Against Candidate B (Unified Framework)

- **Implementation reality check**: The mathematical unification requires showing that existing methods are special cases of a general rule. This is a well-defined mathematical task. The "combined method" is straightforward to implement. The risk is that the combination is either trivially additive (each component helps independently, so combining them is obvious) or that components interfere (CWD mask + cosine schedule may conflict).

- **Reproducibility attack**: The theoretical derivations are reproducible. The experimental comparison inherits from Candidate A's infrastructure. The risk is that the "unified framework" is more of a notational convenience than a genuine theoretical insight -- the reader may say "you just wrote four methods in the same notation."

- **Baseline sanity check**: The combined method needs to beat CWD (the current SOTA simple WD modification). If CWD + cosine schedule is not meaningfully better than CWD alone, the unification does not add algorithmic value. However, the unification has value as theoretical scaffolding even without a winning algorithm.

- **Scope attack**: The unification is comprehensive (4 sub-approaches), which provides good scope. The risk is that the mathematical framework is shallow -- just rewriting existing methods in a common notation without genuine new insight.

- **Verdict**: MODERATE -- The theoretical contribution (unification) is valuable but may be perceived as "just notation." The practical contribution (combined method) has uncertain benefit. This candidate is strengthened by combining it with Candidate A (the benchmark validates the framework).

### Against Candidate C (Fisher-Reweighted WD)

- **Implementation reality check**: Literally one line of code. The `Fishers for Free?` paper (2025) validates that Adam's v_t is a reasonable Fisher proxy. FAdam attempted a related idea. This is entirely doable.

- **Reproducibility attack**: Trivially reproducible. One-line change to AdamW.

- **Baseline sanity check**: **This is the critical test.** FAdam (which uses a more sophisticated Fisher-based correction) was *withdrawn from ICLR 2025*. The reviewers presumably did not find the practical improvements convincing. If FAdam's full correction was not good enough, why would our simpler per-parameter WD reweighting work? The "Fishers for Free?" paper shows the diagonal approximation is decent, but "decent approximation" does not guarantee "meaningful WD improvement." The improvement signal may be too weak to detect above training noise.

- **Scope attack**: One-line modification applied to standard benchmarks. This has narrow scope -- if it does not beat AdamW by a meaningful margin, there is no paper. Unlike CWD (which has a theoretical Pareto-optimality interpretation), Fisher-reweighted WD's theoretical grounding depends on the thermodynamic framework (Caraffa 2026), which is abstract and not yet validated for WD specifically.

- **Verdict**: WEAK -- The FAdam withdrawal is a major red flag. The practical improvement signal is likely too weak. As a *standalone* idea, this is high-risk. However, as a *component* of the unified framework (Candidate B), it adds value: showing that Fisher-reweighted WD is yet another special case of the general rule, and measuring its TEM (thermodynamic efficiency) relative to other methods.


## Phase 4: Refinement

### Dropped Ideas

- **Candidate C (Fisher-Reweighted WD as standalone)** dropped as a lead idea because FAdam's withdrawal signals insufficient practical improvement from Fisher-based WD corrections. The "Fishers for Free?" paper validates the approximation but not the WD benefit. **However**, Fisher-reweighted WD is retained as one method in the comprehensive benchmark (Candidate A) and as a theoretical special case in the unified framework (Candidate B).

### Strengthened Ideas

**Merging Candidates A and B into a single paper:** The strongest paper combines:
1. **Unified mathematical framework** (from Candidate B): Express all WD methods as special cases of `lambda_eff(i,t) = phi(i,t) * lambda_0`, where phi is a per-parameter modulation function. Show that scheduling controls the time dimension of phi, alignment-aware controls the directional dimension, decoupled separates phi from gradient preconditioner, and norm-matched controls the target of phi. This is a genuine structural insight, not just notation.

2. **Comprehensive benchmark** (from Candidate A): Implement all methods in the same codebase with compute-controlled comparison. Use the unified framework to predict which methods should perform similarly (e.g., CWD and cosine schedule both address the "decay-in-wrong-direction" problem but from different angles).

3. **Standardized metrics** (BEM, CSI, AIS): Define precisely, validate that they differentiate methods, and show they explain performance rankings better than accuracy alone.

4. **Systematic visualization**: The killer contribution. Produce comprehensive panels showing per-layer weight norms, alignment cosines, effective learning rates, spectral evolution -- all in a standardized format that makes the differences between methods visually obvious. No existing paper does this systematically.

**Key simplification**: Drop the thermodynamic framing (too abstract, insufficiently validated). Drop Fisher-reweighted WD as a proposed *new* method. Focus on: (1) unified framework as analysis tool, (2) head-to-head benchmark with standardized metrics, (3) visualization toolkit as the primary deliverable. If the combination of CWD + scheduling turns out to outperform both individually, that is a bonus finding, not the main claim.

**Pilot experiment that gives early signal**: CWD vs. AdamW vs. cosine WD schedule on CIFAR-10/ResNet-20, tracking weight norm + alignment cosine + gradient norm. 3 seeds, 15 minutes per run. If the metrics clearly differentiate the three methods (different CSI values, different AIS values), the approach is validated. If all three methods produce indistinguishable metric trajectories, we need to rethink.

### Selected Front-Runner

**"Unified Dynamic Weight Decay: A Framework for Analysis, Comparison, and Diagnosis"** -- combining the theoretical unification, comprehensive benchmark, standardized metrics, and systematic visualization. This is the highest-probability-of-success direction because:

1. The field genuinely needs this comparison paper. Every WD paper benchmarks against AdamW only.
2. The infrastructure exists (open-source repos for all baselines).
3. The standardized metrics are a concrete, citable contribution.
4. The visualization toolkit has standalone value regardless of which method wins.
5. Even if no method dominates, the framework for analyzing *why* methods differ is the contribution.


## Phase 5: Final Proposal

### Title

Unified Dynamic Weight Decay: A Framework for Standardized Analysis, Comparison, and Diagnosis

### Hypothesis

Precisely falsifiable: The four major dynamic WD sub-approaches (scheduling, alignment-aware, decoupled, norm-matched) can be expressed as special cases of a general per-parameter modulation function `lambda_eff(i,t) = phi(w, g, t) * lambda_0`, and a standardized metric suite (Budget Equivalence Metric, Coupling Stability Index, Alignment Informativeness Score) will: (a) differentiate methods that final accuracy alone cannot distinguish, (b) predict method rankings across different architecture/dataset combinations with Kendall tau > 0.7, and (c) reveal that no single existing method dominates all metrics simultaneously.

### Motivation

Weight decay is perhaps the most universally applied technique in deep learning, yet the field suffers from a fragmentation problem: four independent research threads (WD scheduling, alignment-aware WD, decoupled WD, norm-matched WD) each propose improvements using different formulations, different benchmarks, and different evaluation criteria. The practical consequence is that a practitioner cannot make an informed choice between methods -- the papers are simply not comparable.

Specifically:
- CWD (ICLR 2026) reports final loss and accuracy improvements relative to standard decoupled WD.
- SWD (NeurIPS 2023) focuses on gradient norm stability and generalization gap.
- AlphaDecay reports perplexity and spectral density analysis.
- AdamWN describes target-norm control theory without extensive benchmarks.
- gamma^2 scaling (Chou 2025) emphasizes weight norm stability under the Scion optimizer.

There is no way to determine from these papers whether CWD is better than SWD, whether either beats well-tuned AdamW + cosine WD schedule, or whether any of them matter on ImageNet vs. just CIFAR.

Our contribution fills this gap with three components: (1) a unified mathematical framework showing all methods are special cases, (2) a compute-controlled head-to-head benchmark, and (3) standardized metrics and visualization tools.

### Method

**1. Unified Framework: Per-Parameter WD Modulation Function**

The general dynamic WD update is:
```
w_{t+1} = w_t - eta_t * u_t - eta_t * Phi(w_t, u_t, t) * lambda_0 * w_t
```
where u_t is the optimizer update (gradient or preconditioned gradient) and Phi(w, u, t) is the per-parameter modulation matrix (diagonal).

Existing methods correspond to specific choices of Phi:

| Method | Phi(w, u, t) |
|--------|-------------|
| Standard AdamW | I (identity) |
| CWD | diag(sign(w_i * u_i) >= 0) |
| SWD/AdamS | h(||g_t||) * I |
| Cosine WD | cos_schedule(t) * I |
| ADANA (log-time WD) | log_schedule(t) * I |
| AdamWN (target tau) | (1 - tau / ||w||) * I |
| gamma^2 scaling | (eta_t / eta_base) * I |
| AlphaDecay | diag(alpha_layer(l)) |
| Fisher-reweighted | diag(sqrt(v_t(i)) / RMS(sqrt(v_t))) |

This unification is not merely notational: it reveals that:
- Scheduling methods modulate Phi only through t (time-varying scalar).
- Alignment-aware methods modulate Phi through (w, u) (geometry-dependent mask/scalar).
- Norm-matched methods modulate Phi through ||w|| (norm-dependent scalar).
- Per-parameter methods modulate Phi through w_i individually (full diagonal).

The theoretical contribution is proving that these dimensions are *orthogonal* -- they address different failure modes of standard WD, and in principle can be composed. We derive conditions under which compositions are beneficial vs. redundant.

**2. Standardized Evaluation Metrics**

We define three metrics that capture distinct aspects of WD quality:

**Budget Equivalence Metric (BEM)**: For any two methods A and B trained with the same total FLOPs, BEM(A, B) = (metric_A - metric_B) / metric_baseline, where metric is test accuracy or loss, normalized by the baseline (standard AdamW). This enables fair comparison even when papers use different training budgets or schedules.

**Coupling Stability Index (CSI)**: CSI = 1 / (1 + Var_t[||w_t||_2 / ||w_0||_2] / Mean_t[||w_t||_2 / ||w_0||_2]). This measures how stable the weight norm trajectory is relative to its mean. Methods with higher CSI maintain more stable coupling between the optimizer and the WD regularizer. We also propose a per-layer variant that reveals layer-wise instabilities.

**Alignment Informativeness Score (AIS)**: AIS = Corr(cos(w_t, g_t), delta_loss_t), measuring the correlation between gradient-weight alignment and per-step loss improvement. Higher AIS means the alignment signal is actually informative for WD decisions. If AIS is near zero, alignment-aware methods like CWD are not using a meaningful signal.

**3. Comprehensive Benchmark Design**

All methods implemented in a single PyTorch codebase forked from `why-weight-decay`:
- **Architectures**: ResNet-20, VGG-16-BN (CIFAR), ResNet-50 (ImageNet), ViT-S/16 (ImageNet)
- **Datasets**: CIFAR-10, CIFAR-100, ImageNet-1K
- **Methods**: Standard AdamW, CWD-AdamW, SWD/AdamS, Cosine-WD, AdamWN, per-layer AlphaDecay-style, CWD + Cosine (composition test)
- **Protocol**: 3 seeds (42, 123, 456), report mean +/- std. LR individually tuned per method via small grid search (3 values per method). Identical data augmentation, batch size, and total training steps across all methods.
- **Compute fairness**: All methods get the same wall-clock time. Per-step overhead of CWD/SWD is negligible (<1% each).

**4. Systematic Visualization Toolkit**

Produce standardized diagnostic panels for each method:
- Panel 1: Training loss and test accuracy curves (the standard).
- Panel 2: Per-layer weight norm trajectories (reveals norm dynamics differences).
- Panel 3: Per-layer gradient-weight cosine similarity over training (reveals alignment structure).
- Panel 4: Effective learning rate per layer (eta * gradient_norm / weight_norm).
- Panel 5: CSI and AIS evolution over training.
- Panel 6: Phi modulation heatmap (how each method's modulation varies across layers and time).

The toolkit is implemented as a reusable library that accepts logged training statistics and produces publication-quality figures.

### Simplest Version

The absolute minimum experiment that tests the core claim:

1. Implement AdamW, CWD, SWD, Cosine-WD in the same codebase.
2. Train ResNet-20 on CIFAR-10, 3 seeds each.
3. Compute CSI and AIS for each method.
4. If CSI and AIS differentiate the methods (statistically significant difference with p < 0.05), the metric framework is validated.
5. If CSI/AIS rankings correlate with test accuracy rankings (Kendall tau > 0.5), the metrics are useful.

This takes 12 runs x 15 min = 3 GPU-hours. We know within 3 hours whether the core idea works.

### Baselines

1. **Standard AdamW** (the default): Expected CIFAR-10 ResNet-20 test accuracy ~93-94%. The universal baseline.
2. **CWD-AdamW** (current SOTA simple modification): Expected ~0.1-0.3% improvement over AdamW on CIFAR. The method to beat for alignment-aware WD.
3. **SWD/AdamS** (NeurIPS 2023): Expected ~0.3-0.5% improvement over AdamW on CIFAR (less on ImageNet). The method to beat for WD scheduling.
4. **Cosine WD schedule** (common practice): Expected similar to or slightly better than constant WD. The simplest scheduling baseline.
5. **AdamWN** (Loshchilov 2023): No published CIFAR benchmark numbers. We establish the first controlled comparison.
6. **SGD + momentum + constant WD**: The classical baseline. Expected to underperform AdamW on ViT but potentially competitive on ResNet.
7. **CWD + Cosine WD** (composition): No existing benchmark. Tests whether the two approaches compose beneficially.

### Experimental Plan

**Phase 1: Pilot validation** (3 GPU-hours, ~1 hour wall clock)
- CIFAR-10, ResNet-20, {AdamW, CWD, SWD, Cosine-WD}, 3 seeds
- Goal: Validate CSI and AIS differentiate methods
- Kill criterion: If CSI and AIS are indistinguishable across methods, rethink metrics

**Phase 2: CIFAR comprehensive** (18 GPU-hours, ~3 hours wall clock on 8 GPUs)
- CIFAR-10 + CIFAR-100, ResNet-20 + VGG-16-BN, all 7 methods, 3 seeds
- Produce full visualization panels
- Compute all metrics (BEM, CSI, AIS)
- Validate Kendall tau correlation between metrics and accuracy

**Phase 3: ImageNet validation** (72 GPU-hours, ~9 hours wall clock on 8 GPUs)
- ImageNet-1K, ResNet-50, top 4-5 methods from Phase 2, 3 seeds
- Verify metric rankings transfer from CIFAR to ImageNet
- Full visualization panels at scale

**Phase 4: Architecture generalization** (24 GPU-hours, ~3 hours wall clock on 8 GPUs)
- ViT-S/16 on CIFAR-100 and/or ImageNet
- Test whether the unified framework's predictions hold for transformers (LayerNorm, no BN)
- Compare Phi structure across BN and LN architectures

**Total**: ~117 GPU-hours, ~2-3 days wall clock on 8x RTX PRO 6000 Blackwell

### Resource Estimate

| Phase | GPU-hours | Wall clock (8 GPUs) |
|-------|----------|-------------------|
| Pilot | 3 | 1 hour |
| CIFAR full | 18 | 3 hours |
| ImageNet | 72 | 9 hours |
| ViT | 24 | 3 hours |
| **Total** | **~117** | **~16 hours** |

All experiments fit within the project's compute budget (8x RTX PRO 6000 Blackwell, 98GB each). ImageNet experiments will benefit from multi-GPU distributed training. No external API costs -- everything runs locally.

### Risk Assessment

1. **Risk: All methods perform about the same when properly tuned.**
   - Likelihood: MODERATE. CWD gains are 0.1-0.6%, which may be within noise on CIFAR with 3 seeds.
   - Mitigation: (a) Use enough seeds (3 is minimum, consider 5 if variance is high). (b) Focus the contribution on the *diagnostic framework* rather than "method X beats method Y." A well-documented negative result ("these methods are equivalent when controlled for compute") is still publishable if the metrics and visualization are rigorous. (c) The visualization toolkit has standalone value.

2. **Risk: CSI and AIS do not differentiate methods.**
   - Likelihood: LOW. Weight norm trajectories and alignment cosines are known to differ across WD methods (the `why-weight-decay` paper and CWD paper both show distinct trajectories).
   - Mitigation: The pilot (Phase 1) tests this cheaply. If the metrics fail, we have fallback metrics: gradient norm ratio, effective LR stability, spectral condition number.

3. **Risk: Unified framework is perceived as "just notation."**
   - Likelihood: MODERATE. The risk is that writing everything as Phi(w, u, t) is seen as trivial.
   - Mitigation: (a) Prove substantive theorems: e.g., that scheduling and alignment-aware modulation are *orthogonal* (compose beneficially), while alignment-aware and norm-matched are *redundant* (CWD already controls norms as a side effect). (b) The composition experiment (CWD + Cosine WD) is the key test of whether the framework predicts real behavior.

4. **Risk: ImageNet experiments take too long or fail.**
   - Likelihood: LOW (8x 98GB GPUs with distributed training should be fine).
   - Mitigation: Start with single-GPU ImageNet-100 subset for quick validation, then scale to full ImageNet.

5. **Risk: Difficulty in reproducing exact numbers from existing papers.**
   - Likelihood: MODERATE. Different random seeds, data augmentation details, and training recipes can cause 0.1-0.3% accuracy differences.
   - Mitigation: Report our own baselines (AdamW in our codebase) rather than citing numbers from other papers. All comparisons are within-codebase.

### Novelty Claim

The novelty is **not** a new WD algorithm. The novelty is:

1. **Unified Phi-framework**: The first mathematical framework that expresses all four WD sub-approaches as special cases of a per-parameter modulation function, with proofs of orthogonality/redundancy between approaches. This tells practitioners which combinations to try and which to skip.

2. **Standardized metrics (BEM, CSI, AIS)**: The first quantitative metrics specifically designed for evaluating and comparing dynamic WD methods. These are concrete, computable, and citable.

3. **Compute-controlled benchmark**: The first head-to-head comparison of all major WD variants (CWD, SWD, AdamWN, cosine schedule, per-layer adaptive) on the same codebase with identical training budgets. This alone fills a significant gap.

4. **Systematic visualization toolkit**: The first comprehensive diagnostic panel for WD methods, enabling visual comparison of weight norm trajectories, alignment structures, effective learning rates, and modulation patterns across methods and architectures.

5. **Composition analysis**: The first study of whether WD modifications compose (CWD + scheduling, alignment + norm-matching). The framework predicts which compositions should work; the experiments validate.

The paper type is "analysis + benchmark + diagnostic framework" -- similar in spirit to "Benchmarking Neural Network Robustness to Common Corruptions" (Hendrycks & Dietterich, ICLR 2019), which was highly cited not because it proposed a new method but because it provided the field with a standardized evaluation framework.
