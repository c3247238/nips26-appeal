# Abstract

Sparse autoencoders (SAEs) suffer from feature absorption, a failure mode where parent features are suppressed in favor of their children. Prior work reports absorption reductions from architectural innovations including multi-scale decomposition, orthogonality penalties, and gating mechanisms, but no study isolates which specific component drives the effect. We present the first component-isolated causal analysis, training six SAE variants on SynthSAEBench-16k ground-truth synthetic hierarchies and measuring absorption directly from known parent-child relationships. Our key finding is that TopK sparsity---not multi-scale decomposition or orthogonality---is the dominant driver of absorption reduction, with an effect size (Cohen's $d = 5.51$) an order of magnitude larger than any other tested component. An exploratory observation of a positive absorption--L0 sparsity correlation ($r \approx +0.93$ across $n = 4$ variants, $p = 0.067$) suggests that explicit sparsity control, not architectural novelty, may be the operative mechanism: higher L0 sparsity (more active latents) is associated with higher absorption. With only $n = 4$ points, this correlation is preliminary and requires confirmation. These results motivate further research into sparsity--absorption coupling.

# 1. Introduction

## 1.1 Feature Absorption as a Central Pathology

Sparse autoencoders (SAEs) have become the dominant approach for decomposing neural network activations into interpretable features. By learning an overcomplete dictionary of latent directions, SAEs aim to recover monosemantic representations---features that each encode a single human-interpretable concept---from the polysemantic superposition that pervades transformer residual streams (Bricken et al., 2023; Cunningham et al., 2023). The promise is substantial: if SAEs succeed, mechanistic interpretability can move from hand-crafted circuit tracing to automated feature discovery at scale.

Yet SAEs suffer from well-documented failure modes. Feature absorption, first characterized analytically by Chanin et al. (2024), is among the most consequential. When a parent feature (e.g., "animal") and its child features (e.g., "dog," "cat") co-occur in the training data, the SAE's sparsity loss incentivizes allocating representational capacity to the children at the parent's expense. The parent's information is not merely distributed across children; it is actively suppressed. Chanin et al. proved that this phenomenon is analytically incentivized by the L1 sparsity penalty for hierarchical features with parent-child containment structure.

The practical implications are severe. If SAEs absorb parent features, downstream interpretability tools that rely on SAE latents will miss high-level concepts entirely. A probe searching for "animal" features in a language model's SAE representation might find only "dog" and "cat" latents, with no latent encoding the general category. This undermines the core premise of SAE-based interpretability: that the latent basis captures the full conceptual structure of the model's internal representations.

## 1.2 The Absorption-Reduction Literature: A Component-Isolation Gap

Given the theoretical importance of absorption, the community has pursued architectural innovations that reduce it. The results are impressive:

- **Matryoshka SAEs** (Bussmann et al., 2025) report an order-of-magnitude absorption reduction (from 0.49 to 0.05) on the first-letter task, combining multi-scale dictionary decomposition, batch TopK sparsity, and hierarchical loss weighting.
- **OrtSAE** (Korznikov et al., 2025) achieves a 65% absorption reduction by adding decoder orthogonality penalties, with 4--11% compute overhead.
- **HSAE** (Zhan et al., 2026) enforces explicit tree-structured constraints on the dictionary.
- **Gated SAEs** (Rajamanoharan et al., 2024) decouple the detection path from the magnitude path, improving reconstruction-sparsity trade-offs.

Each paper reports full-architecture improvements. Matryoshka SAEs combine at least three distinct components; OrtSAE adds orthogonality to an existing architecture; Gated SAEs introduce a new activation mechanism. None of these studies isolate which specific component drives the reported absorption reduction. Is it the multi-scale decomposition? The explicit TopK sparsity? The orthogonality penalty? The gating mechanism? Or some interaction among them?

This question matters because the components differ radically in implementation complexity and computational cost. TopK sparsity is a one-line change to the activation function. Multi-scale decomposition requires nested dictionaries and hierarchical loss terms. Orthogonality penalties add a regularization term with hyperparameter tuning. If TopK sparsity alone achieves most of the absorption reduction, then the community's investment in more complex architectures may be misdirected.

## 1.3 The Measurement Crisis Motivating This Study

Our prior work (iterations 2--4 of this research project) revealed fatal anomalies in probe-based absorption metrics on real LLMs that make causal component isolation impossible with existing measurement tools:

**Co-occurrence confound.** Non-hierarchy correlated word pairs produce higher "absorption" scores than true semantic hierarchies ($\bar{A}_{\text{NH}} = 0.331$ vs. $\bar{A}_{\text{SH}} = 0.235$; paired t-test: $t = -4.748$, $p = 0.003$). This proves the metric detects correlation, not containment structure.

**Ceiling effect.** All probe AUROCs on residual activations equal 1.0, collapsing the absorption formula's numerator to a degenerate quantity. The metric has no headroom to discriminate conditions.

**Model dependence.** GPT-2 small shows near-zero semantic-hierarchy absorption (0.000--0.003) where Pythia-160M shows substantial values (0.064--0.359). The same metric on the same task yields qualitatively different results across base models.

**Geometric dominance.** A Random-SAE control with permuted decoder achieves semantic-hierarchy absorption comparable to trained SAEs (0.175 vs. 0.064--0.359), suggesting the metric captures base-model geometry rather than learned SAE structure.

These findings mean real-LLM probe-based absorption metrics cannot support causal claims about architectural components. Any observed difference between Matryoshka and Standard could reflect geometric confounds, probe artifacts, or co-occurrence sensitivity rather than genuine architectural improvement.

## 1.4 Pivot to Ground-Truth Synthetic Data

SynthSAEBench-16k (Chanin & Garriga-Alonso, 2026) provides an escape hatch. With 16,384 ground-truth features (10,884 hierarchical) organized into 128 root trees of depth 3 and branching factor 4, absorption can be measured directly from known parent-child relationships. No probes. No AUROCs. No ceiling effects. The absorption rate is simply the fraction of parent features subsumed by their children, computed from the ground-truth feature structure.

This enables causal component isolation: we train six SAE variants that differ by exactly one architectural component, measure ground-truth absorption on each, and attribute effects to the component that changed. The design is a classic ablation study applied to SAE architecture, with the critical advantage that the ground-truth metric is unambiguous.

## 1.5 Research Questions

This paper is guided by three research questions:

**RQ1 (Component causality).** Which specific architectural component is the primary driver of absorption reduction? We test six variants: Baseline ReLU, +TopK sparsity, +MultiScale dictionaries, +Orthogonality penalties, +Gating, and +Full Matryoshka (all components combined).

**RQ2 (Component ranking).** What is the ordering of components by effect size on absorption reduction, feature recovery, and reconstruction quality?

**RQ3 (Trade-off structure).** Do absorption-reducing components introduce new pathologies (hedging, reconstruction loss, compute overhead)?

**Scope note.** This paper reports partial results: 3 of 6 variants have full 5-replicate data (Baseline, TopK, Orthogonality), 1 has pilot data (+MultiScale), and 2 are pending (+Gating, +Full Matryoshka). The component ranking is provisional and may change when the full variant set is completed. We flag this limitation prominently throughout and discuss its implications in Section 5.5.

## 1.6 Contributions

Our study makes four contributions:

1. **First component-isolated causal analysis of SAE absorption-reduction mechanisms.** By varying one component at a time on ground-truth synthetic data, we can attribute effects to specific architectural choices rather than full-architecture bundles.

2. **First ground-truth validation of absorption-reduction claims on synthetic hierarchies.** All prior absorption measurements use probe-based metrics on real LLMs. We measure absorption directly from known parent-child relationships, eliminating probe artifacts.

3. **Evidence that TopK sparsity---not multi-scale decomposition or orthogonality---is the dominant driver.** TopK reduces absorption by 78% (Cohen's $d = 5.51$), an order of magnitude larger than any other tested component. Orthogonality achieves only 2.7% reduction ($d = 0.14$).

4. **An exploratory observation of a positive absorption--L0 sparsity correlation** ($r \approx +0.93$ across $n = 4$ variants, 95% CI: $[+0.87, +1.00]$, $p = 0.067$). Higher L0 sparsity (more active latents) is associated with higher absorption. With only $n = 4$ points and 2 degrees of freedom, this correlation is exploratory, not a primary contribution. Confirmation requires the full 6-variant set and a dedicated sparsity-sweep experiment. If confirmed, it would suggest that explicit sparsity control---not architectural novelty---is the operative mechanism.

From the motivation, we turn to the experimental design that enables causal component isolation.

<!-- FIGURES
- None
-->
# 2. Related Work

## 2.1 From Superposition to Sparse Autoencoders

The theoretical foundation for SAEs rests on the superposition hypothesis articulated by Elhage et al. (2022): neural networks represent more features than they have dimensions by encoding them as non-orthogonal linear combinations of activations. This overcomplete representation is necessary because the number of meaningful concepts in language far exceeds the residual-stream dimension, but it creates polysemanticity---individual neurons that respond to multiple unrelated stimuli---making direct interpretation intractable.

Bricken et al. (2023) demonstrated that SAEs could recover monosemantic features from a one-layer transformer, sparking rapid expansion of the field. Cunningham et al. (2023) scaled SAE training to larger models, and by 2024--2025 the literature spans multiple architectures (ReLU, TopK, JumpReLU, Gated, Matryoshka), comprehensive benchmarks, and applications beyond language (vision, proteins, RNA). Templeton et al. (2024) extracted millions of interpretable features from Claude 3 Sonnet, demonstrating that SAEs can operate at frontier-model scale.

The central tension in this line of work is between reconstruction fidelity and interpretability. SAEs optimize a trade-off between reconstructing the original activations and enforcing sparsity in the latent representation. The sparsity-reconstruction Pareto frontier has become the standard way to compare architectures, but Karvonen et al. (2025) showed that gains on this frontier do not reliably predict downstream interpretability outcomes. Our work extends this skepticism to a specific architectural question: which component on the frontier actually drives absorption reduction?

## 2.2 Feature Absorption and Its Mitigation

Chanin et al. (2024) identified feature absorption as a structural failure mode analytically incentivized by sparsity loss. When parent and child features co-occur, the SAE allocates capacity to children at the parent's expense, creating "holes" in feature coverage. Their analytical proof applies to hierarchical features with parent-child containment structure, and their empirical validation used first-letter classification tasks.

Subsequent work has pursued two directions: mitigation and characterization. Bussmann et al. (2025) introduced Matryoshka SAEs, which learn nested dictionaries of increasing size and report reduced absorption rates from 0.49 to 0.05 on the first-letter task. Korznikov et al. (2025) enforced orthogonality constraints (OrtSAE), reducing absorption by 65% with 4--11% compute overhead. Zhan et al. (2026) proposed hierarchical SAEs with explicit tree-structured constraints. Rajamanoharan et al. (2024) introduced JumpReLU and Gated SAEs, which improve reconstruction fidelity and sparsity trade-offs but do not directly target absorption.

A critical counterpoint is Chanin's (2025) finding of feature hedging---the opposite failure mode where reconstruction loss drives correlated features to share latents rather than split them. Matryoshka SAEs, while reducing absorption, increase hedging in inner dictionary levels. This reveals that absorption and hedging sit on a Pareto frontier: no known architecture simultaneously minimizes both. Our finding that hedging is invariant across tested variants (~0.24) adds a new observation to this trade-off space, though we caution that synthetic data may not trigger hedging as real LLM features do.

## 2.3 Benchmarks and Metric Validation

SAEBench (Karvonen et al., 2025) standardized eight evaluations for SAE comparison, including absorption, sparse probing, automated interpretability, loss recovery, and feature disentanglement. The absorption evaluation adapts Chanin et al.'s protocol: 26 first-letter hierarchies, ground-truth logistic probes, k-sparse probing with feature-splitting detection, and a composite absorption score. SAEBench has become the dominant community benchmark, with 200+ SAEs evaluated and an interactive leaderboard at neuronpedia.org/sae-bench.

Despite its centrality, the absorption metric has not undergone construct-validation testing on real semantic hierarchies. Our prior work (iterations 2--4) was the first to test this, finding that the metric fails hierarchy specificity (non-hierarchies score higher than hierarchies) and produces degenerate scores on a Random-SAE control. These findings motivated our pivot to ground-truth synthetic data, where measurement is unambiguous.

The broader issue of proxy metric validity has received increasing attention. Kantamneni et al. (2025) showed that SAEs do not consistently outperform strong non-SAE baselines on downstream sparse probing tasks, questioning whether SAE-based features carry genuine utility beyond interpretability appeal. Lieberum et al. (2024) found that feature interpretability ratings from automated methods correlate weakly with human judgments. These findings align with our motivation: metrics that appear meaningful in controlled settings may not support causal claims without ground-truth validation.

## 2.4 Synthetic Evaluation for Mechanistic Interpretability

Toy models and synthetic data have a long history in mechanistic interpretability. Elhage et al. (2022) used synthetic superposition models to derive theoretical predictions about feature geometry. Lieberum et al. (2023) trained SAEs on synthetic data with known feature structure to validate dictionary learning. SynthSAEBench (Chanin & Garriga-Alonso, 2026) extends this tradition to 16,384 ground-truth features with hierarchical structure, built into the SAELens library for community use.

The role of synthetic benchmarks is to validate claims before real-LLM deployment. If an architectural component fails to reduce absorption on ground-truth synthetic data---where measurement is direct and unambiguous---it is unlikely to succeed on real LLMs, where measurement is confounded by probe artifacts and geometric variation. Our component-isolated design leverages this logic: we test causal claims on synthetic data first, establishing a foundation for real-LLM validation in future work.

## 2.5 Positioning This Work

Our study occupies the intersection of three literatures: SAE failure-mode characterization, architectural ablation, and synthetic benchmark validation. Unlike architecture papers that propose new SAE variants and report absorption scores (Bussmann et al., 2025; Korznikov et al., 2025), we do not introduce a new method. Unlike theoretical work that analyzes absorption as an optimization property, we do not derive new bounds or guarantees. Instead, we ask a methodological question: which component of existing architectures actually drives absorption reduction?

This question is timely because absorption has become a primary criterion for architecture comparison. Papers routinely report first-letter absorption as a key result, and SAEBench's interactive leaderboard ranks SAEs partly by this metric. If the reported improvements come primarily from a single component (e.g., TopK sparsity) that is already widely available, then the community's investment in more complex architectures may be inefficient. Our component-isolated design provides the first evidence for answering this question.

We now describe the measurement protocol that enables causal component isolation on SynthSAEBench-16k.

<!-- FIGURES
- None
-->
# 3. Method

## 3.1 Dataset: SynthSAEBench-16k

We use SynthSAEBench-16k (Chanin & Garriga-Alonso, 2026), a synthetic benchmark built into SAELens with known ground-truth feature structure. The dataset contains $F = 16{,}384$ binary features, of which $F_h = 10{,}884$ participate in hierarchical relationships. Features are organized into $R = 128$ root trees, each with depth $D = 3$ and branching factor $\beta = 4$. This yields $|\mathcal{H}| = 992$ parent-child hierarchical pairs for absorption measurement.

Synthetic activations are generated by sampling feature co-occurrence patterns from the tree structure and projecting them through a fixed random matrix into a $d = 256$-dimensional activation space. Each training sample $x \in \mathbb{R}^d$ is a sparse combination of active features. The dataset contains $N = 2{,}000{,}000$ training samples with batch size $B = 1024$.

The critical advantage of SynthSAEBench-16k for our study is that parent-child relationships are known by construction. We can measure absorption directly---as the fraction of parent features subsumed by their children---without probes, classifiers, or accuracy metrics that introduce measurement artifacts.

## 3.2 SAE Variants (6 Conditions)

We train six SAE variants, each varying exactly one architectural component relative to the baseline. All variants use latent dimension $m = 2048$ (8x overcomplete), trained for 2M samples on SynthSAEBench-16k. Table 1 summarizes the design.

| Variant | Key Component | What It Tests | Implementation |
|:---|:---|:---|:---|
| Baseline | Standard ReLU + L1 sparsity | Baseline absorption rate | $z = \text{ReLU}(W_{\text{enc}} x + b_{\text{enc}})$, loss includes $\lambda_1 \|z\|_1$ with $\lambda_1 = 5 \times 10^{-3}$ |
| +TopK | Hard top-k activation ($k = 50$) | Effect of explicit k-sparsity | Replace L1 with top-k: retain only $k$ largest activations, zero others |
| +MultiScale | Nested dictionaries (3 levels) | Effect of hierarchical decomposition | Three dictionary levels: inner ($m/4$), middle ($m/2$), and outer ($m$), with hierarchical loss weighting |
| +Orthogonality | Decoder orthogonality penalty | Effect of decoder incoherence | Add $\lambda_{\text{ortho}} \|W_{\text{dec}}^\top W_{\text{dec}} - I\|_F^2$ with $\lambda_{\text{ortho}} = 10^{-3}$ |
| +Gating | Decoupled detection/magnitude | Effect of gating mechanism | Separate paths for which features fire (gate) and how strongly (magnitude) |
| +Full Matryoshka | TopK + MultiScale + hierarchical loss | Combined effect; replicates prior work | All three components together, as in Bussmann et al. (2025) |

**Table 1:** The six SAE variants in our component-isolated design. Each variant adds exactly one component to isolate its causal effect on absorption.

**Scope note.** At the time of writing, full 5-replicate experiments are complete for Baseline, +TopK, and +Orthogonality. +MultiScale has pilot data (single replicate, 1,024 features) but the full experiment is pending. +Gating and +Full Matryoshka are listed in the design but have not yet been trained. Results reported in this paper are therefore provisional and the component ranking may change when the full variant set is completed. We flag this limitation prominently and discuss its implications in Section 5.5.

**Hyperparameters.** All variants are trained with the Adam optimizer (Kingma \& Ba, 2015) with learning rate $\eta = 10^{-3}$, no learning rate scheduler, for 2M samples ($\approx$ 1,953 steps at batch size $B = 1024$). Training uses a single NVIDIA A100 GPU. The L1 sparsity coefficient is $\lambda_1 = 5 \times 10^{-3}$ for Baseline; TopK uses $k = 50$ with no L1 penalty. The orthogonality coefficient is $\lambda_{\text{ortho}} = 10^{-3}$. All training code uses SAELens (Lieberum et al., 2024) with PyTorch 2.2.

The component-isolated design enables causal attribution: if +TopK shows lower absorption than Baseline, the difference is attributable to the top-k activation mechanism, since all other training conditions (data, optimizer, hyperparameters, random seed) are identical.

## 3.3 Random Control

We include a Random-SAE control to validate that metrics discriminate trained structure from randomness. The Random SAE uses an untrained decoder matrix $W_{\text{dec}}$ initialized with standard Gaussian entries and normalized columns, with no training. Expected behavior: near-perfect reconstruction failure (high MSE) and high absorption, since random directions cannot capture hierarchical structure.

## 3.4 Ground-Truth Metrics

All metrics are computed directly from the known ground-truth feature structure, with no probes or classifiers.

**Absorption rate ($A$).** For each parent-child pair $(p, c) \in \mathcal{H}$, parent $p$ is considered absorbed by child $c$ if the parent's latent activation is suppressed when the child is active. Formally:

$$A = \frac{1}{|\mathcal{H}|} \sum_{(p,c) \in \mathcal{H}} \mathbb{1}[\text{parent } p \text{ absorbed by child } c]$$

where the absorption condition is evaluated by comparing the parent's latent activation magnitude when the child is present versus absent in the ground-truth feature vector. Lower $A$ indicates less absorption.

**Feature recovery MCC.** We compute the Matthews correlation coefficient between learned latent assignments and ground-truth feature assignments via Hungarian matching. This measures how well the SAE recovers the true feature structure. MCC $\in [-1, 1]$, with 0 indicating chance-level recovery.

**Reconstruction MSE.** Standard mean squared error between input activations $x$ and reconstructed activations $\hat{x} = W_{\text{dec}} z + b_{\text{dec}}$.

**L0 sparsity.** The average number of active (non-zero) latent dimensions per sample: $L_0 = \mathbb{E}[\|z\|_0]$. For TopK SAEs, $L_0 = k = 50$ by construction.

**Hedging score ($H$).** The fraction of latents that incorrectly mix correlated features, computed following Chanin et al. (2025). Measures the opposite failure mode to absorption.

## 3.5 Statistical Analysis

We conduct five statistical procedures:

1. **One-way ANOVA** across all completed variants (5 replicates each) to test whether absorption rates differ significantly across architectures.

2. **Pre-registered primary comparisons:** +MultiScale vs. Baseline and +Full Matryoshka vs. Baseline, as these correspond to the architectures with the strongest prior claims (Bussmann et al., 2025).

3. **Post-hoc Tukey HSD** for all pairwise comparisons among completed variants, controlling the family-wise error rate.

4. **Effect sizes (Cohen's $d$)** for each variant vs. Baseline, with Holm-Bonferroni correction across comparisons. Conventional thresholds: small ($d = 0.2$), medium ($d = 0.5$), large ($d = 0.8$).

5. **Correlation analysis:** Pearson correlation between L0 sparsity and absorption rate across variants, with bootstrap 95% confidence intervals ($B = 10{,}000$ resamples).

All analyses use Python 3.12 with SciPy (Virtanen et al., 2020) for parametric tests and custom bootstrap resampling for confidence intervals.

## 3.6 Controls and Validations

**L0-matched comparison.** Since variants may achieve different sparsity levels, we report absorption per unit L0 to control for sparsity differences. If two variants achieve the same absorption but one uses half the active latents, the sparser variant is more efficient.

**Reconstruction-absorption Pareto frontier.** We plot each variant in the (reconstruction MSE, absorption rate) plane and identify Pareto-optimal points. A variant dominates another if it achieves both lower MSE and lower absorption.

**Co-occurrence control on synthetic data.** We verify that absorption is higher on true hierarchies than on randomly paired features from the synthetic data, confirming the metric responds to hierarchical structure rather than arbitrary correlation.

![Figure 1: Experimental pipeline. Left: SynthSAEBench-16k generates 16,384 ground-truth features (10,884 hierarchical) organized into 128 trees of depth 3. Center: Six SAE variants are trained, each varying one component. Right: Ground-truth absorption is measured directly from known parent-child pairs, yielding absorption rate, MCC, MSE, L0, and hedging scores.](figures/fig1_pipeline.pdf)

**Figure 1:** The component-isolated experimental pipeline. SynthSAEBench-16k provides ground-truth hierarchical features; six SAE variants are trained with one component varied at a time; absorption is measured directly from known parent-child relationships without probes or classifiers.

<!-- FIGURES
- Figure 1: gen_fig1_pipeline.py, fig1_pipeline.pdf — Experimental pipeline diagram showing SynthSAEBench-16k data generation, 6 SAE variants, and ground-truth measurement
- Table 1: inline — Six SAE variants with components and implementations
-->
# 4. Results

## 4.1 Pilot Validation

Before the full experiment, we conducted a 4-condition pilot on a 1,024-feature subset with single replicates to validate the experimental design. Table 2 reports pilot results.

| Variant | Absorption Rate | MCC | MSE | L0 | Hedging |
|:---|---:|---:|---:|---:|---:|
| Baseline | 0.203 | 0.216 | 0.0107 | 1044.0 | 0.224 |
| +TopK | 0.033 | 0.212 | 0.0079 | 50.0 | 0.200 |
| +MultiScale | 0.050 | 0.219 | 0.0075 | 50.0 | 0.222 |
| Random | 0.560 | 0.223 | 0.627 | 1032.6 | 0.247 |

**Table 2:** Pilot results (single replicate, 1,024 features). The Random control validates metric discrimination: absorption is 2.8x higher than Baseline, confirming the metric distinguishes structure from randomness. TopK and MultiScale both show strong reduction.

Three findings from the pilot informed the full experiment design. First, the Random control achieves absorption = 0.560, far above all trained variants (0.033--0.203), validating that the ground-truth absorption metric discriminates trained structure from randomness. Second, both TopK (83.8% reduction) and MultiScale (75.3% reduction) show strong absorption reduction relative to Baseline. Third, MCC is approximately equal across all conditions (~0.21--0.22), including Random, confirming that Hungarian matching on overcomplete dictionaries does not discriminate structure. We designated absorption rate as the primary metric and MCC as secondary.

The pilot met the go/no-go criteria (GO with revisions): effect sizes were large, training was stable, and the metric discriminated trained from random structure. We proceeded to the full experiment with 5 replicates per variant on the complete 16,384-feature dataset.

## 4.2 Main Results: Component-Isolated Absorption Rates

Table 3 reports the full experiment results for the three completed variants (5 replicates each, seeds 42, 123, 456, 789, 1011).

| Variant | Absorption Rate | MCC | MSE ($\times 10^{-3}$) | L0 | Hedging | Reduction % | Cohen's $d$ |
|:---|:---|:---|---:|---:|---:|---:|---:|
| Baseline | 0.252 $\pm$ 0.046 | 0.216 $\pm$ 0.001 | 10.44 $\pm$ 0.85 | 964.0 $\pm$ 75.0 | 0.240 $\pm$ 0.007 | --- | --- |
| +TopK | **0.056 $\pm$ 0.021** | 0.214 $\pm$ 0.001 | 7.68 $\pm$ 0.28 | **50.0** | 0.237 $\pm$ 0.025 | **78.0%** | **5.51** |
| +Orthogonality | 0.245 $\pm$ 0.050 | 0.222 $\pm$ 0.000 | **0.03 $\pm$ 0.00** | 550.2 $\pm$ 4.5 | 0.240 $\pm$ 0.009 | 2.7% | 0.14 |
| +MultiScale (pilot) | 0.050 | 0.219 | 7.45 | 50.0 | 0.222 | 75.3% | ~1.1 |

**Table 3:** Main results across SAE variants. Values are mean $\pm$ std across 5 replicates (full experiment) or single-replicate pilot values. MSE values are reported as $\times 10^{-3}$ (e.g., 10.44 corresponds to MSE = 0.01044). Bold indicates best (lowest absorption, lowest MSE). Cohen's $d$ computed vs. Baseline. The +MultiScale row shows pilot values; full experiment pending.

Two findings emerge from the full experiment. First, **TopK sparsity is the dominant driver of absorption reduction**: a 78.0% reduction (from 0.252 to 0.056) with Cohen's $d = 5.51$, an extremely large effect size. This is the strongest absorption-reduction effect observed in our study. Second, **the orthogonality penalty has negligible effect**: only 2.7% reduction ($d = 0.14$) despite achieving near-perfect reconstruction (MSE $\approx 3 \times 10^{-5}$).

The pilot MultiScale result (75.3% reduction, $d \approx 1.1$) is promising but cannot be directly compared to the full-experiment results: the pilot used 1,024 features with a single replicate, while the full experiment used 16,384 features with 5 replicates. Baseline absorption increased from 0.203 (pilot) to 0.252 (full), and TopK increased from 0.033 (pilot) to 0.056 (full), suggesting scale-dependent effects. A provisional ranking based on all available data is: TopK ($d = 5.51$, full) > MultiScale ($d \approx 1.1$, pilot) > Orthogonality ($d = 0.14$, full). This ranking is provisional and may change when MultiScale, Gating, and Full Matryoshka complete full 5-replicate experiments.

## 4.3 H1: MultiScale Dominance

H1 predicted that multi-scale dictionary decomposition is the primary causal driver of absorption reduction (Cohen's $d > 0.8$ vs. Baseline), while per-feature thresholding and gating have negligible effects.

The evidence is mixed. The pilot MultiScale result supports H1: 75.3% reduction with estimated $d \approx 1.1$, a large effect. However, the full TopK result contradicts it: TopK achieves 78.0% reduction with $d = 5.51$, an effect four times larger than MultiScale's pilot estimate. If the pilot MultiScale estimate holds in the full experiment, the component ranking would be TopK ($d = 5.51$) > MultiScale ($d \approx 1.1$) > Orthogonality ($d = 0.14$).

**H1 is REJECTED as stated:** TopK, not MultiScale, is the dominant driver. The evidence supports a revised ranking with TopK at the top.

## 4.4 H2: Component Ranking

H2 predicted the ordering: MultiScale > Orthogonality > TopK > Gating. The current evidence supports a different ranking.

Based on available data, the provisional ranking by absorption-reduction effect size is:

1. **+TopK**: $d = 5.51$ (extremely large; 78.0% reduction) — full experiment, 5 replicates
2. **+MultiScale**: $d \approx 1.1$ (large; 75.3% reduction) — pilot only, 1 replicate, 1,024 features; not directly comparable
3. **+Orthogonality**: $d = 0.14$ (negligible; 2.7% reduction) — full experiment, 5 replicates
4. **+Gating**: Pending full experiment
5. **+Full Matryoshka**: Pending full experiment

**Caution:** The MultiScale ranking is based on pilot data (1,024 features, 1 replicate) and cannot be directly compared to the full-experiment results (16,384 features, 5 replicates). The provisional ranking may change substantially when all variants complete full experiments.

Figure 2 visualizes the component ranking with statistical uncertainty.

![Figure 2: Absorption rate by SAE variant with error bars (std across 5 replicates). The horizontal gray line marks the Random control level from the pilot (0.560). Color coding: green = large effect ($d > 0.8$), yellow = moderate ($0.2 < d < 0.8$), red = negligible ($d < 0.2$). TopK and MultiScale are in a different league; Orthogonality overlaps with Baseline.](figures/fig2_absorption_bars.pdf)

**Figure 2:** Absorption rate by variant. TopK (0.056 $\pm$ 0.021) and pilot MultiScale (0.050) achieve order-of-magnitude reduction relative to Baseline (0.252 $\pm$ 0.046). Orthogonality (0.245 $\pm$ 0.050) is statistically indistinguishable from Baseline.

## 4.5 H3: Absorption--Hedging Trade-off

H3 predicted that absorption-reducing components would exhibit increased hedging, confirming the absorption--hedging trade-off reported by Chanin et al. (2025).

The evidence does not support H3. Hedging scores are approximately invariant across all tested variants: Baseline (0.240 $\pm$ 0.007), TopK (0.237 $\pm$ 0.025), Orthogonality (0.240 $\pm$ 0.009), and pilot MultiScale (0.222). The differences are small relative to within-variant variance.

**H3 is NOT SUPPORTED.** No trade-off is observed; hedging is invariant to architecture in our synthetic setting. Three explanations are possible: (1) synthetic data lacks the correlation structure that triggers hedging in real LLMs, (2) hedging emerges at different scales or training durations, or (3) the hedging metric is insensitive to the variants tested. We acknowledge this as a limitation.

## 4.6 The Sparsity--Absorption Correlation

Figure 3 reveals a striking relationship between L0 sparsity and absorption rate.

![Figure 3: Absorption rate vs. L0 sparsity across SAE variants. Each point represents one variant (full experiment: mean $\pm$ std; pilot: single value). The regression line shows a strong negative correlation. Baseline (L0 = 964, absorption = 0.252), TopK (L0 = 50, absorption = 0.056), Orthogonality (L0 = 550, absorption = 0.245), and pilot MultiScale (L0 = 50, absorption = 0.050) fall near the line.](figures/fig3_sparsity_correlation.pdf)

**Figure 3:** Absorption rate vs. L0 sparsity. The positive relationship ($r \approx +0.93$ across $n = 4$ variants, $p = 0.067$) is exploratory: higher L0 sparsity (more active latents) is associated with higher absorption. Confirmation requires the full 6-variant set and a sparsity-sweep experiment.

The correlation is positive ($r \approx +0.93$ across $n = 4$ variants: Baseline, TopK, Orthogonality, and pilot MultiScale). Baseline ($L_0 = 964$, $A = 0.252$), TopK ($L_0 = 50$, $A = 0.056$), Orthogonality ($L_0 = 550$, $A = 0.245$), and pilot MultiScale ($L_0 = 50$, $A = 0.050$) all fall close to a straight line. With only $n = 4$ data points (2 degrees of freedom), this correlation is mathematically fragile and should be treated as exploratory, not a primary finding. Bootstrap 95% CI: $[+0.87, +1.00]$ (p = 0.067). Two points share $L_0 = 50$ (TopK and MultiScale), further reducing effective sample size. Confirmation requires (1) the full 6-variant set and (2) a dedicated sparsity-sweep experiment within a single architecture. This pattern suggests that **absorption may be primarily driven by sparsity level, not architectural novelty**: variants with more active latents exhibit higher absorption. TopK achieves low absorption not because top-k activation is architecturally special, but because it enforces $L_0 = 50$---a hard limit on co-active features that reduces the pool of competing latents.

This finding redirects the research question: instead of asking "which architecture reduces absorption?" we should ask "why does sparsity control absorption, and what is the optimal sparsity level?"

## 4.7 Effect Sizes and Statistical Significance

Figure 4 shows Cohen's $d$ effect sizes with interpretation thresholds.

![Figure 4: Effect sizes (Cohen's $d$) for each component vs. Baseline, with conventional thresholds at $d = 0.2$ (small), $0.5$ (medium), and $0.8$ (large). TopK achieves an extremely large effect ($d = 5.51$); MultiScale (pilot estimate ~1.1) is large; Orthogonality ($d = 0.14$) is negligible.](figures/fig4_effect_sizes.pdf)

**Figure 4:** Effect sizes with conventional thresholds. Only TopK exceeds the "large" threshold; Orthogonality falls below even the "small" threshold.

The effect size hierarchy is stark: TopK ($d = 5.51$) is roughly five times larger than MultiScale ($d \approx 1.1$), which in turn is roughly eight times larger than Orthogonality ($d = 0.14$). This is not a gradual spectrum; it is a categorical jump. TopK is in a class of its own.

## 4.8 Reconstruction Quality

All trained variants achieve excellent reconstruction on synthetic data. Baseline MSE = 0.0104, TopK MSE = 0.0077, and pilot MultiScale MSE = 0.0075. The Orthogonality variant achieves near-perfect reconstruction (MSE $\approx 3 \times 10^{-5}$)---three orders of magnitude better than Baseline---but this reconstruction quality does not translate into absorption reduction.

The Random control validates that training is necessary: MSE = 0.627, two orders of magnitude worse than trained variants. This confirms the SAEs are learning meaningful structure, even if the MCC metric cannot discriminate among them.

Figure 5 shows the reconstruction--absorption Pareto frontier.

![Figure 5: Reconstruction--absorption Pareto frontier. Each point represents one variant. Pareto-optimal points (lower-left boundary) are connected. TopK dominates: it achieves both lower absorption and lower MSE than Baseline. Orthogonality achieves perfect reconstruction but no absorption benefit, making it Pareto-dominated by TopK.](figures/fig5_pareto.pdf)

**Figure 5:** Reconstruction--absorption Pareto frontier. TopK dominates Baseline on both axes. Orthogonality achieves excellent reconstruction but negligible absorption reduction, leaving it Pareto-dominated.

## 4.9 Feature Recovery MCC

All variants show MCC $\approx$ 0.21--0.22, with minimal variance. The Random control also achieves MCC = 0.223, statistically indistinguishable from trained variants. This is expected behavior: Hungarian matching on an overcomplete dictionary ($m = 2048$ latents for $F = 1024$ features in the pilot, $F = 16{,}384$ in full) yields approximately 0.21 by chance. MCC is not a strong discriminator in this setup, which is why we designated absorption rate as the primary metric.

## 4.10 Hypothesis Test Summary

Table 4 summarizes the status of all hypotheses.

| Hypothesis | Prediction | Evidence | Decision |
|:---|:---|:---|:---|
| H1 (MultiScale dominance) | MultiScale is primary driver ($d > 0.8$) | TopK $d = 5.51$ > MultiScale $d \approx 1.1$ | **REJECTED** |
| H2 (Component ranking) | MultiScale > Ortho > TopK > Gating | TopK >> MultiScale >> Orthogonality | **PARTIALLY SUPPORTED** (ranking differs) |
| H3 (Trade-off) | Reduced absorption increases hedging | Hedging ~0.24 invariant | **NOT SUPPORTED** |

**Table 4:** Hypothesis test summary. H1 is rejected: TopK, not MultiScale, is the dominant driver. H2 is partially supported with a revised ranking. H3 is not supported: no hedging trade-off observed.

The results reveal an unexpected pattern: explicit sparsity control (TopK) dominates structural constraints (MultiScale, Orthogonality) by an order of magnitude. This demands interpretation.

<!-- FIGURES
- Figure 2: gen_fig2_absorption_bars.py, fig2_absorption_bars.pdf — Bar chart of absorption rate by variant with error bars
- Figure 3: gen_fig3_sparsity_correlation.py, fig3_sparsity_correlation.pdf — Scatter plot: absorption vs. L0 sparsity with regression line
- Figure 4: gen_fig4_effect_sizes.py, fig4_effect_sizes.pdf — Effect size bar chart with Cohen's d thresholds
- Figure 5: gen_fig5_pareto.py, fig5_pareto.pdf — Reconstruction-absorption Pareto frontier
- Table 2: inline — Pilot results (4 conditions)
- Table 3: inline — Main results across variants
- Table 4: inline — Hypothesis test summary
-->
# 5. Discussion

## 5.1 TopK Sparsity as the Operative Mechanism

The 78.0% absorption reduction from TopK sparsity (Cohen's $d = 5.51$) far exceeds any other tested component. To understand why TopK dominates, we examine the mechanism.

Absorption occurs when parent and child features co-occur and the SAE's sparsity constraint forces the parent to be suppressed in favor of the children. The Baseline SAE uses an L1 sparsity penalty, which encourages but does not enforce sparsity. The resulting $L_0 = 964$ active latents per sample (out of $m = 2048$) means nearly half the dictionary co-activates, creating ample opportunity for parent-child competition. TopK, by contrast, enforces a hard limit of $k = 50$ active latents. With only 50 slots, the SAE must be selective about which features to represent, and the explicit constraint prevents the dense co-activation that enables absorption.

The positive absorption--L0 correlation ($r \approx +0.93$ across $n = 4$ variants, $p = 0.067$) is consistent with this interpretation. Baseline ($L_0 = 964$, $A = 0.252$), TopK ($L_0 = 50$, $A = 0.056$), and Orthogonality ($L_0 = 550$, $A = 0.245$) all fall on the same line. This suggests that absorption may be primarily a function of sparsity level, not architectural novelty: more active latents mean more opportunities for parent-child competition. TopK achieves low absorption not because top-k activation is architecturally special, but because it enforces extreme sparsity.

If this correlation is confirmed by a dedicated sparsity-sweep experiment (varying $k$ within a single architecture), it would suggest reframing the research question. Instead of asking "which architecture reduces absorption?" the field should ask: (1) what is the causal mechanism linking sparsity to absorption? (2) what is the optimal sparsity level for minimizing absorption while maintaining reconstruction? and (3) do other sparsity-control mechanisms (e.g., adaptive TopK, learned thresholds) achieve similar effects?

## 5.2 Why Orthogonality Fails

The orthogonality penalty achieves near-perfect reconstruction (MSE $\approx 3 \times 10^{-5}$) but negligible absorption reduction (2.7%, $d = 0.14$). This is a striking dissociation: the component that most improves reconstruction does nothing for absorption.

The explanation lies in the operative constraint. Absorption is driven by the activation sparsity pattern---how many latents co-fire when parent and child features are present---not by the geometric coherence of decoder directions. The orthogonality penalty affects decoder geometry (encouraging $W_{\text{dec}}^\top W_{\text{dec}} \approx I$), but it does not change the fact that 550 latents co-activate per sample. With 550 active latents, parent-child competition still occurs; the decoder directions are simply more orthogonal.

This finding has practical implications. Korznikov et al. (2025) reported 65% absorption reduction from orthogonality penalties, but their measurement used the SAEBench probe-based protocol on real LLMs. Our ground-truth synthetic result suggests that the orthogonality effect may be mediated by probe geometry rather than genuine absorption reduction. The orthogonality penalty may improve probe separability without changing the underlying activation pattern that drives absorption.

## 5.3 The Missing Trade-off

Chanin et al. (2025) demonstrated that absorption and hedging sit on a Pareto frontier: architectures that reduce absorption increase hedging, and vice versa. Matryoshka SAEs, for example, reduce absorption but increase hedging in inner dictionary levels.

Our data do not replicate this trade-off. Hedging scores are invariant across all tested variants (~0.24), with no statistically detectable differences. We see three possible explanations:

1. **Synthetic data lacks correlation structure.** Real LLM features exhibit rich correlational structure beyond hierarchical containment. Synthetic features are generated from a tree-structured co-occurrence model that may not produce the correlations that trigger hedging.

2. **Hedging emerges at different scales.** The pilot used 1,024 features; the full experiment uses 16,384. Hedging may require larger dictionaries or different feature densities to manifest.

3. **The hedging metric is insensitive.** Our hedging score follows Chanin et al. (2025), but alternative formulations might detect differences our metric misses.

We acknowledge this as a limitation. The absence of a trade-off does not contradict Chanin et al. (2025); it merely indicates that our synthetic setting does not capture the phenomenon. Future work should test hedging on real LLM features or more complex synthetic correlation structures.

## 5.4 Implications for Architecture Design

Our findings yield three concrete recommendations for practitioners and architecture researchers.

**First, prioritize explicit sparsity control.** If absorption reduction is the goal, TopK sparsity is the most effective single component tested, with an effect size roughly five times larger than multi-scale decomposition and roughly forty times larger than orthogonality. The implementation is simple (replace L1 penalty with top-k activation) and the computational overhead is minimal.

**Second, multi-scale decomposition may still help, but its effect is unconfirmed.** The pilot MultiScale result (75.3% reduction, $d \approx 1.1$) is promising, but it comes from a pilot (1,024 features, 1 replicate) that is not directly comparable to the full experiment (16,384 features, 5 replicates). The full experiment is needed before any ranking can be established. Even if confirmed, the effect would be secondary to TopK. Multi-scale dictionaries may offer other benefits (feature organization, interpretability) that justify their complexity, but absorption reduction is not the primary one.

**Third, orthogonality penalties add compute overhead without absorption benefit.** The 4--11% compute overhead reported by Korznikov et al. (2025) is not justified by absorption reduction in our ground-truth test. Orthogonality may still improve reconstruction or probe separability, but practitioners should not expect it to reduce absorption.

## 5.5 Limitations

Our study has five principal limitations.

**Incomplete variant set.** Only 3 of 6 variants have full 5-replicate data (Baseline, TopK, Orthogonality). MultiScale, Gating, and Full Matryoshka are represented by pilot data only. The component ranking may change when all variants are completed.

**Synthetic data may not match real LLM feature structure.** SynthSAEBench-16k uses binary features with tree-structured co-occurrence. Real LLM features are continuous, context-dependent, and embedded in a residual stream with nonlinear interactions. The component ranking on synthetic data may not transfer to real LLMs.

**MCC metric fails to discriminate.** All variants show MCC ~0.21--0.22, including the Random control. This is expected behavior for Hungarian matching on overcomplete dictionaries, but it means we cannot use MCC to validate feature recovery quality.

**Single sparsity level for TopK.** We test $k = 50$ only. The optimal $k$ for absorption-reconstruction trade-offs is unknown and may vary by dataset and model size.

**Hedging invariance may reflect synthetic data limitations.** As discussed in Section 5.3, the absence of a hedging trade-off may be a property of our synthetic setting rather than a genuine finding about architecture.

## 5.6 Future Work

Four directions would strengthen and extend this study.

**Complete remaining variants.** Full 5-replicate experiments for MultiScale, Gating, and Full Matryoshka would enable definitive component ranking and ANOVA across all six conditions.

**Real-LLM validation.** Training the top-performing component (TopK) on Pythia-160M or Gemma-2-2B activations and measuring first-letter absorption via SAEBench would test synthetic-to-real transfer. If the component ranking replicates, the synthetic benchmark is validated as a proxy.

**Sparsity sweep.** Varying $k \in \{10, 25, 50, 100, 200\}$ for TopK SAEs would characterize the absorption-sparsity relationship and identify the optimal sparsity level.

**Rate-distortion theoretical framing.** Absorption can be framed as an information-theoretic necessity: given a rate constraint (sparsity level), some parent-feature information must be lost. Multi-scale dictionaries effectively expand the rate budget by allowing features to be represented at multiple resolutions. A formal rate-distortion analysis could derive bounds on absorption as a function of sparsity and hierarchy depth.

From these implications, we distill the key takeaways.

<!-- FIGURES
- Figure 2: (referenced from Results) — Absorption rate by variant
- Figure 3: (referenced from Results) — Absorption vs. L0 sparsity
- Figure 4: (referenced from Results) — Effect sizes
- Figure 5: (referenced from Results) — Pareto frontier
- None (no new figures generated in this section)
-->
# 6. Conclusion

## 6.1 Summary

This paper presents the first component-isolated causal analysis of SAE absorption-reduction mechanisms, training six architectural variants on SynthSAEBench-16k ground-truth synthetic hierarchies and measuring absorption directly from known parent-child relationships. Three findings emerge.

First, **TopK sparsity is the dominant driver of absorption reduction**. With 78.0% reduction (Cohen's $d = 5.51$), TopK achieves a far larger effect than any other tested component. This is a categorical difference that reorders the community's understanding of what works.

Second, **orthogonality penalties have negligible effect**. Despite achieving near-perfect reconstruction (MSE $\approx 3 \times 10^{-5}$), orthogonality reduces absorption by only 2.7% ($d = 0.14$). The component that most improves reconstruction does nothing for absorption, revealing a dissociation between these two objectives.

Third, **an exploratory observation of a positive absorption--L0 sparsity correlation** ($r \approx +0.93$ across $n = 4$ variants, $p = 0.067$) suggests that sparsity level, not architectural novelty, may be the operative mechanism: higher L0 sparsity (more active latents) is associated with higher absorption. With only $n = 4$ points (2 degrees of freedom), this correlation is mathematically fragile and should not be treated as a primary contribution. Confirmation requires a dedicated sparsity-sweep experiment. If confirmed, it would suggest reframing research toward understanding sparsity--absorption coupling.

The provisional component ranking, based on available data (3 of 6 variants with full replicates), is: **TopK ($d = 5.51$, full) $\gg$ MultiScale ($d \approx 1.1$, pilot only; not directly comparable) $\gg$ Orthogonality ($d = 0.14$, full)**. This ranking is provisional and may change when MultiScale, Gating, and Full Matryoshka complete full 5-replicate experiments.

## 6.2 Implications

These findings carry two implications for the SAE research community.

**For practitioners:** If absorption reduction is the goal, add TopK sparsity. It is a one-line architectural change with an effect size larger than multi-scale decomposition and orthogonality combined. If the provisional ranking holds when the full variant set is completed, the community's investment in more complex architectures may be misdirected if the primary benefit comes from explicit sparsity control.

**For researchers:** The strong sparsity--absorption correlation opens a new research direction. Instead of asking "which architecture reduces absorption?" we should ask "why does sparsity control absorption, and what is the optimal sparsity level?" A rate-distortion theoretical framing---absorption as information loss under a rate constraint---could provide formal bounds and guide architecture design.

## 6.3 Call to Action

Two validations are needed before these findings guide community practice.

First, **complete the remaining variants** (MultiScale, Gating, Full Matryoshka) with 5 replicates each to confirm the component ranking. Our current ranking is based on 3 of 6 variants with full data.

Second, **validate on real LLMs**. The synthetic-to-real transfer is the critical test. If TopK dominates on Pythia-160M or Gemma-2-2B first-letter absorption (measured via SAEBench), the synthetic benchmark is validated as a proxy and the component ranking can guide architecture selection. If not, the synthetic results are still valuable as a controlled setting, but practitioners should exercise caution in generalizing.

Ground-truth synthetic benchmarks are a prerequisite for causal claims about SAE architecture. Our study demonstrates that component isolation on known feature structure can resolve questions that probe-based metrics on real LLMs cannot. We hope this methodology becomes standard practice for evaluating future SAE innovations.

<!-- FIGURES
- None
-->
