# Innovator Perspective

## Phase 1: Literature Survey

### Key Papers Found

1. **Chanin et al., 2024/2025. "A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders." arXiv:2409.14507 (NeurIPS 2025)** — First systematic definition of feature absorption; introduced absorption rate metric; proved via toy models that hierarchy + sparsity optimization inevitably causes absorption; validated on hundreds of LLM SAEs. Critical limitation: ablation-dependent metric limited to layers 0-17.

2. **Hu et al., 2025. "Measuring Sparse Autoencoder Feature Sensitivity." arXiv:2509.23717** — Proposed feature sensitivity as a new evaluation dimension; found many interpretable features have poor sensitivity; sensitivity declines with SAE width across 7 variants up to 1M features. Uses LLM-generated similar texts to measure activation consistency.

3. **Bereska et al., 2025. "Superposition as Lossy Compression: Measure with Sparse Autoencoders and Connect to Adversarial Vulnerability." arXiv:2512.13568 (TMLR)** — Formalized superposition through nested rate-distortion problems; defined "effective features" via Shannon entropy on SAE activations; connected superposition degree to adversarial robustness. Strong correlation with ground truth in toy models.

4. **Engels, Smith & Tegmark, 2024/2025. "Decomposing The Dark Matter of Sparse Autoencoders." arXiv:2410.14670** — Decomposed SAE reconstruction error into linearly predictable and nonlinear components; found nonlinear error is responsible for proportional downstream cross-entropy degradation; nonlinear error stays constant as SAE width scales, serving as a fundamental lower bound.

5. **Smith et al., 2025. "Negative Results for Sparse Autoencoders On Downstream Tasks and Deprioritising SAE Research." DeepMind Safety Research** — Critical negative results showing SAEs underperform simple baselines on downstream tasks; SAEs are better for discovery than intervention; led to deprioritization of SAE research in some teams.

6. **Karvonen et al., 2025. "SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability." arXiv:2503.09532 (ICML 2025)** — Eight-metric benchmark (absorption, sparse probing, auto-interpretability, RAVEL, unlearning, SCR, TPP, CE loss) on 200+ SAEs; revealed proxy metrics do not predict practical performance.

7. **Bussmann et al., 2025. "Learning Multi-Level Features with Matryoshka Sparse Autoencoders." arXiv:2503.17547** — Nested dictionaries trained simultaneously; reduced absorption from 0.49 to 0.05; smaller dicts learn general features, larger dicts specialize. ~50% computational overhead.

8. **Luo et al., 2026. "Building a Structured Feature Forest with Hierarchical Sparse Autoencoders." arXiv:2602.11881** — HSAE jointly trains series of SAEs with explicit parent-child relationships; substantially outperforms baselines on absorption, especially at larger sizes.

9. **Wu et al., 2025. "Interpreting and Steering LLMs with Mutual Information-based Explanations on Sparse Autoencoders." arXiv:2502.15576** — Uses mutual information to create explanations for SAE features, directly linking information-theoretic measures to SAE interpretability and steering.

10. **Han et al., 2025. "Causal Interpretation of Sparse Autoencoder Features in Vision." arXiv:2509.00749** — Addresses correlational vs. causal distinction in SAE features; uses Effective Receptive Field with input-attribution methods to identify patches that causally drive activations.

11. **Marks et al., 2025/2026. "A Unified Theory of Sparse Dictionary Learning in Mechanistic Interpretability." arXiv:2512.05534** — Theoretical framework: piecewise biconvexity and spurious minima; feature anchoring improves recovery across architectures; explains why simple linear encoders are hard to outperform.

12. **Xu et al., 2025. "Model Unlearning via Sparse Autoencoder Subspace Guided Projections." ACL Anthology (EMNLP 2025)** — SSPU uses SAE features to drive targeted parameter updates for interpretable unlearning; reduces harmful knowledge accuracy by 3.22% vs. strongest baseline.

### Landscape Summary

The SAE field in 2025-2026 is at an inflection point. On one hand, there has been explosive growth in architectures (TopK, JumpReLU, Matryoshka, HSAE, OrtSAE, AdaptiveK) and benchmarks (SAEBench, SynthSAEBench, CE-Bench). On the other hand, a growing body of negative results questions whether SAEs can reliably support downstream interventions.

Three critical tensions define the current landscape:

**Tension 1: Discovery vs. Intervention.** The DeepMind negative results paper and follow-up work ("Use SAEs to Discover Unknown Concepts, Not to Act on Known Concepts") have reframed SAEs as discovery tools rather than intervention tools. This creates a gap: if SAE features are unreliable for steering and circuit editing, what is their practical value beyond visualization?

**Tension 2: Reconstruction vs. Behavior.** Multiple papers ("Dark Matter," SAEBench, "Sparse Autoencoders Do Not Find Canonical Features") show that reconstruction quality (MSE, explained variance) is a poor proxy for downstream behavioral preservation. The nonlinear error component disproportionately harms downstream performance, yet SAE training does not explicitly minimize it.

**Tension 3: Ablation-dependence vs. Scalability.** The primary absorption metric (Chanin et al.) requires causal ablation experiments that only work for early layers (<= layer 17 in Gemma 2 2B). Deep attention layers have moved information to final token positions, making ablation effects disappear. This means we have no reliable way to measure absorption in the deeper layers where the most interesting features may reside.

The cross-domain opportunity I see is this: neuroscience has long grappled with a similar problem — how to distinguish "real" feature selectivity from epiphenomenal co-activation. The solution in neuroscience was not better measurement of individual neurons, but understanding the circuit context and inhibitory mechanisms that shape selectivity. I believe a similar shift is needed in SAE research: from measuring individual features to understanding the information-theoretic and circuit-level context that determines feature reliability.

## Phase 2: Initial Candidates

### Candidate A: Information-Theoretic Absorption Index — A Training-Free Unified Quality Score

- **Hypothesis**: The degree of feature absorption in an SAE can be quantified training-free by measuring the information-theoretic divergence between encoder and decoder weight distributions for each latent, yielding an "Absorption Index" that correlates with ground-truth ablation-based absorption rates and predicts downstream intervention failure.
- **Cross-domain insight**: From **rate-distortion theory** (Bereska et al., 2025) and the **information bottleneck principle** (Tishby et al., 2000). In lossy compression, the encoder and decoder form a matched pair that minimizes distortion at a given rate. When absorption occurs, the encoder "lies" about feature presence (fails to activate the parent) while the decoder "covers up" (reconstructs the parent direction from the child latent). This creates an information-theoretic mismatch — the encoder and decoder are no longer a matched rate-distortion pair.
- **Evidence for**: (1) Chanin et al. explicitly note that "absorption leads to an asymmetric pattern in the encoder and decoder of the SAE" and suggest this as a future training-free detection direction. (2) The "Decoder-Free SAE" paper (arXiv:2601.06478) identifies that "asymmetry between encoder and decoder is characteristic of absorption." (3) Bereska et al.'s rate-distortion framework provides the mathematical machinery to quantify this mismatch via KL divergence or mutual information between encoder and decoder representations. (4) Engels et al.'s "dark matter" decomposition shows that nonlinear error (which should correlate with absorption) is responsible for downstream CE degradation.
- **Novelty estimate**: 8/10 — The observation of encoder-decoder asymmetry exists in the literature but has never been formalized into a quantitative, training-free metric. The information-theoretic framing is new. The connection to predicting downstream intervention failure is unexplored.

### Candidate B: Absorption-Sensitivity Duality — A Joint Measurement Framework for Feature Reliability

- **Hypothesis**: Feature absorption (directional stealing by child features) and feature sensitivity (inconsistent activation on similar inputs) are dual manifestations of the same underlying pathology: the SAE's inability to faithfully represent hierarchical structure under sparsity constraints. A joint measurement framework combining both metrics will reveal a "reliability manifold" where features cluster into distinct quality regimes, and the position on this manifold predicts causal efficacy for downstream interventions.
- **Cross-domain insight**: From **quantum mechanics complementarity** and **signal processing joint time-frequency analysis**. Just as position and momentum are Fourier duals that together characterize a quantum state, absorption and sensitivity are dual metrics that together characterize feature reliability. Absorption measures spatial/directional fidelity (does the feature point in the right direction?), while sensitivity measures temporal/activation fidelity (does it fire consistently when it should?). Neither alone is sufficient; their joint distribution reveals the true state.
- **Evidence for**: (1) Hu et al. found sensitivity is "weakly correlated with existing metrics (frequency: rho=-0.06, decoder cosine similarity: rho=0.06, interpretability: rho=0.24)" — suggesting it captures a genuinely novel dimension. (2) Chanin et al. found absorption increases with higher sparsity and wider SAEs. (3) No paper has jointly measured both on the same feature set. (4) The literature gap analysis explicitly identifies this as Gap 5: "Their quantitative relationship is unclear." (5) The "reliability manifold" concept connects to the "effective features" metric from Bereska et al. — features with poor absorption-sensitivity profiles may correspond to low-effective-feature regions.
- **Novelty estimate**: 7/10 — The individual metrics exist. Joint measurement is proposed as a gap but not executed. The duality framing and reliability manifold concept are new. The prediction of causal efficacy from the joint metric is novel.

### Candidate C: Cross-Layer Absorption Propagation — How Early-Layer Feature Absorption Corrupts Downstream Circuit Completeness

- **Hypothesis**: Feature absorption in early layers propagates through the network via the residual stream, causing systematic corruption of downstream feature representations that compounds with depth. This propagation follows a predictable mathematical structure (approximable as a multiplicative error accumulation) and can be detected by measuring the "absorption footprint" in downstream layer SAEs without requiring deep-layer ablation.
- **Cross-domain insight**: From **error propagation in deep learning** (similar to how batch normalization papers analyzed internal covariate shift) and **epidemiological compartmental models** (SIR models). Just as an infectious disease spreads through a population via contact networks, absorption "infects" downstream representations via the residual stream connectivity. The "absorption footprint" in layer L+1 can be predicted from the absorption rate in layer L plus the layer's mixing properties (attention pattern entropy).
- **Evidence for**: (1) RouteSAE (EMNLP 2025) shows features peak at different layers, with low-level features in shallow layers and high-level features in deep layers — suggesting cross-layer propagation matters. (2) "Rethinking Circuit Completeness" (Chen et al., 2025) identifies OR gates as a root cause of circuit incompleteness — absorption creates exactly this OR-gate structure where parent features are distributed across multiple child latents. (3) Marks et al.'s "Sparse Feature Circuits" (2024) established SAE features as circuit nodes, but did not study how absorption affects circuit completeness. (4) The literature gap analysis identifies Gap 3: "how absorption affects sparse feature circuits, model editing, unlearning, and concept erasure" as underexplored. (5) "Decomposing The Dark Matter" shows nonlinear error stays constant with width — suggesting it propagates rather than being corrected by wider SAEs.
- **Novelty estimate**: 9/10 — Cross-layer analysis of absorption is entirely unexplored. The mathematical structure of propagation is novel. The connection to circuit completeness (Chen et al.'s OR-gate framework) is a fresh cross-pollination. The epidemiological analogy provides intuitive explanatory power.

## Phase 3: Self-Critique

### Against Candidate A: Information-Theoretic Absorption Index

- **Prior work attack**: The "Decoder-Free SAE" paper (arXiv:2601.06478) goes further than just noting asymmetry — it proposes eliminating the decoder entirely. If decoder-free SAEs become standard, the encoder-decoder asymmetry metric becomes obsolete. However, decoder-free SAEs are very new (January 2026) and have not been validated at scale. The vast majority of existing pretrained SAEs (Gemma Scope, GPT-2 SAEs, etc.) have both encoder and decoder, so the metric remains relevant for the training-free analysis constraint.
- **Methodological attack**: Computing KL divergence between encoder and decoder weights assumes a specific probabilistic model over weights that may not be justified. The encoder and decoder weights are deterministic parameters, not distributions. A more defensible approach would use cosine similarity or mutual information between activations (not weights), but this requires data — it's not purely weight-based. The metric may also conflate absorption with other encoder-decoder asymmetries (e.g., dead latents, feature splitting).
- **Theoretical attack**: The rate-distortion analogy is elegant but may be superficial. In standard rate-distortion theory, the encoder and decoder are designed jointly to minimize distortion. In SAEs, the encoder and decoder are trained jointly but the asymmetry arises from the sparsity constraint, not from a rate-distortion mismatch. The structural correspondence may not hold mathematically.
- **Scalability attack**: The metric needs to be validated against ground-truth ablation-based absorption rates. But ablation only works for early layers. For deep layers, there is no ground truth, so the metric cannot be validated. This limits its applicability to the very layers where absorption detection is most needed.
- **Verdict**: MODERATE — The core insight is sound and the gap is real. The main weaknesses are: (1) the need to carefully define the information-theoretic quantity to avoid conflating with other phenomena, and (2) the validation problem for deep layers. Both are addressable with careful experimental design.

### Against Candidate B: Absorption-Sensitivity Duality

- **Prior work attack**: The SAEBench paper already measures multiple metrics on the same SAEs (including absorption and sensitivity-related proxies). However, they do not measure the *Hu et al.* sensitivity metric specifically, nor do they analyze the joint distribution. The "weak correlation" finding from Hu et al. suggests the metrics are indeed capturing different things, but a joint analysis may simply reveal two independent quality dimensions rather than a meaningful duality.
- **Methodological attack**: Measuring sensitivity requires LLM-generated similar texts (Hu et al. use GPT-4.1), which is computationally expensive and introduces dependence on the LLM's quality. For a training-free analysis project, this is a significant burden. The joint measurement may also be dominated by the sensitivity metric's noise, swamping the absorption signal.
- **Theoretical attack**: The quantum mechanics analogy is a metaphor, not a mathematical correspondence. Position and momentum are Fourier duals with a precise uncertainty relation. Absorption and sensitivity are not Fourier duals — they are just two different quality metrics. Calling them "dual" may be intellectually appealing but theoretically vacuous unless a precise mathematical relationship can be derived.
- **Scalability attack**: The "reliability manifold" concept requires high-dimensional visualization or clustering, which may not scale to SAEs with millions of features. The computational cost of computing both metrics for all features may exceed the project's time budget.
- **Verdict**: MODERATE — The joint measurement is genuinely valuable and addresses a clear gap. The "duality" framing is a metaphor that needs to be either mathematized or dropped. The computational cost of sensitivity measurement is a real concern but can be mitigated with sampling.

### Against Candidate C: Cross-Layer Absorption Propagation

- **Prior work attack**: The RouteSAE paper (EMNLP 2025) already studies cross-layer feature dynamics, but from a different angle — it routes activations from multiple layers into a shared feature space. The "Skip Transcoders" paper (Paulo et al., 2025) also studies cross-layer transformations. However, neither studies absorption propagation specifically. The "Crosscoders with Structured Priors" (LessWrong, 2025) uses graph Laplacian priors to trace concept propagation across layers, but does not connect to absorption.
- **Methodological attack**: Measuring "absorption footprint" in downstream layers requires a training-free absorption metric for deep layers — which brings us back to the problem that Candidate A tries to solve. Without a deep-layer absorption metric, the propagation study cannot be executed. This creates a dependency: Candidate C requires Candidate A (or some other deep-layer metric) to work.
- **Theoretical attack**: The epidemiological analogy is vivid but may mislead. Absorption is not an infectious process — it is a structural property of each layer's SAE optimization. The residual stream carries activations, not absorption. A more accurate model would treat each layer's absorption as an independent optimization outcome, with correlation across layers due to shared hierarchical structure in the data, not propagation.
- **Scalability attack**: This requires analyzing SAEs across all layers of a model (e.g., 27 layers of Gemma 2), computing absorption-related metrics for each, and fitting a propagation model. The computational cost may exceed the 1-hour-per-experiment budget. The analysis also requires circuit-tracing infrastructure that may not be readily available.
- **Verdict**: MODERATE — The idea is highly novel but methodologically demanding. The dependency on a deep-layer absorption metric is a critical path risk. The "propagation" framing may need to be reframed as "correlation across layers" rather than causal propagation.

## Phase 4: Refinement

### Dropped Ideas

- None of the three candidates were rated WEAK. All have MODERATE ratings with addressable weaknesses.

### Strengthened Ideas

- **Candidate A (Information-Theoretic Absorption Index)**: Reframed to avoid the weight-distribution KL divergence problem. Instead of comparing weight distributions, compute the **mutual information between encoder activations and decoder reconstructions** for each latent on a held-out dataset. Absorbed latents will show anomalous MI patterns: the decoder reconstructs a feature direction that the encoder rarely activates. This is data-dependent but still training-free (no retraining). The metric is: `Absorption_Index(i) = 1 - MI(encode_i(x), decode_i(z)) / H(encode_i(x))`, where low values indicate absorption. Validation: compare with Chanin et al.'s ablation-based absorption rate on layers 0-17 where ground truth exists.

- **Candidate B (Absorption-Sensitivity Duality)**: Dropped the quantum mechanics metaphor. Reframed as a **"Feature Reliability Biplot"** — a 2D scatter plot of absorption index (x-axis) vs. sensitivity (y-axis) for each feature, with density contours showing clustering into quality regimes. The key insight is not "duality" but **"complementary failure modes"**: high-absorption features have directional errors (wrong latent represents the feature), while low-sensitivity features have temporal errors (right latent fires inconsistently). A feature can have one, both, or neither. The "reliability score" is a simple combination: `reliability = (1 - normalized_absorption) * sensitivity`. To address computational cost, sensitivity is measured on a stratified sample of features (e.g., 100 features per SAE) rather than all features.

- **Candidate C (Cross-Layer Absorption Propagation)**: Reframed from "propagation" to **"cross-layer absorption correlation"**. The hypothesis is revised: layers that share similar hierarchical structure in their inputs will show correlated absorption patterns, and the strength of this correlation can be predicted from the layer's attention pattern entropy (high entropy = more mixing = less correlation). This removes the causal propagation claim while retaining the cross-layer insight. The dependency on Candidate A is acknowledged and turned into a strength: the project will first develop the training-free absorption index (Candidate A), then use it to study cross-layer patterns (Candidate C).

### Additional Evidence Found

- **"Superposition as Lossy Compression" (Bereska et al., 2025)**: Provides strong theoretical support for the information-theoretic framing. The "effective features" metric (F = minimum neurons for interference-free encoding) directly connects to absorption — absorbed features are a form of interference where one latent encodes multiple hierarchical features. The paper's Shannon entropy measure on SAE activations can be adapted per-latent for the absorption index.

- **"Causal Interpretation of Sparse Autoencoder Features in Vision" (Han et al., 2025)**: Supports the need for causal validation of SAE features. Their Effective Receptive Field method can be adapted to text SAEs by measuring the "effective context" that causally drives a feature's activation, which may be smaller for absorbed features (since the child latent only fires in specific contexts).

- **"Sparse Autoencoders Do Not Find Canonical Features" (ICLR 2025)**: Shows that different SAE training runs learn different feature sets, with implications for reproducibility. This strengthens the case for training-free metrics — if features are not canonical, training-dependent metrics are unreliable.

### Selected Front-Runner

**Candidate A: Information-Theoretic Absorption Index** is selected as the front-runner for the following reasons:

1. **Directly addresses the most critical gap**: The inability to measure absorption in deep layers without ablation is the single biggest methodological limitation in the field. Solving it would have immediate practical value for all SAE researchers.

2. **Aligns perfectly with project constraints**: The project is explicitly training-free. Candidate A requires no SAE retraining, only analysis of existing pretrained SAEs (Gemma Scope, GPT-2 SAEs).

3. **Theoretical grounding**: The rate-distortion and information-theoretic framing provides rigorous mathematical foundations, not just heuristic engineering.

4. **Scalable validation**: Can be validated on layers 0-17 where ablation works, then extrapolated to deeper layers with confidence intervals.

5. **Enables downstream work**: Once the index exists, it enables Candidate B (joint analysis with sensitivity) and Candidate C (cross-layer correlation) as natural follow-ups within the same project.

6. **Novelty**: While encoder-decoder asymmetry has been noted, no prior work has formalized it into a quantitative, validated, training-free metric with theoretical justification.

## Phase 5: Final Proposal

### Title

An Information-Theoretic Absorption Index for Sparse Autoencoders: A Training-Free Metric for Detecting Feature Absorption Across All Layers

### Hypothesis

The degree of feature absorption in a sparse autoencoder latent can be quantified training-free by measuring the normalized mutual information between that latent's encoder activation and decoder reconstruction on held-out data. Latents with abnormally low normalized mutual information (indicating the decoder reconstructs a feature direction that the encoder rarely activates) correspond to absorbed features. This "Absorption Index" correlates strongly with ground-truth ablation-based absorption rates in early layers (where ablation is feasible) and can be extrapolated to deep layers (where ablation fails) to enable the first systematic absorption analysis across all model layers.

### Motivation

Feature absorption is one of the most critical pathologies in sparse autoencoders, undermining their reliability for mechanistic interpretability. The current gold-standard absorption metric (Chanin et al.) requires causal ablation experiments that only work for early layers (<= layer 17 in Gemma 2 2B) because deep attention layers have moved information to final token positions, making ablation effects disappear. This means we have no reliable way to measure absorption in the deeper layers where the most abstract and interesting features reside.

The inability to measure absorption at depth has cascading consequences:
- **Safety**: Absorbed safety-relevant features (e.g., deception indicators) may evade detection in deep layers, creating hidden risks.
- **Circuit analysis**: Automated circuit-finding methods miss critical pathways when features are distributed across unexpected latents.
- **Architecture comparison**: We cannot fairly compare absorption-mitigating architectures (Matryoshka, HSAE, OrtSAE) at depth.
- **Quality assurance**: Researchers using pretrained SAEs (like Gemma Scope) have no way to screen for absorbed features without expensive ablation experiments.

A training-free, theoretically-grounded absorption metric would solve all of these problems.

### Method

**Step 1: Define the Absorption Index**

For each latent i in an SAE, compute on a held-out dataset D:

```
Absorption_Index(i) = 1 - MI(encode_i(x), decode_i(z_i)) / H(encode_i(x))
```

Where:
- `encode_i(x)` is the encoder activation of latent i on input x
- `decode_i(z_i)` is the decoder reconstruction contribution of latent i (i.e., `z_i * W_dec[i]` projected onto the feature direction)
- `MI` is mutual information estimated via k-NN or binning
- `H` is Shannon entropy

The normalization by entropy makes the index comparable across latents with different activation frequencies. An index near 1 indicates high absorption (encoder rarely fires, decoder reconstructs the direction); near 0 indicates no absorption (encoder and decoder are well-matched).

**Step 2: Validate against ground truth**

On layers 0-17 of Gemma 2 2B (where Chanin et al.'s ablation-based absorption rate works), compute the Pearson correlation between the Absorption Index and the ground-truth absorption rate. Target: r > 0.7. Also validate on GPT-2 small SAEs using the sae-spelling first-letter classification task.

**Step 3: Extrapolate to deep layers**

Apply the validated Absorption Index to layers 18-26 of Gemma 2 2B (where ablation is infeasible). Report absorption rates across all layers for the first time. Test whether absorption increases monotonically with depth (as hierarchical features become more abstract) or follows a different pattern.

**Step 4: Cross-architecture comparison**

Apply the index to compare absorption across SAE architectures (TopK, JumpReLU, ReLU) on the same model layer, controlling for width and sparsity. This enables the first fair, training-free comparison of absorption mitigation strategies.

**Step 5: Connect to downstream impact**

Measure the correlation between Absorption Index and: (a) feature sensitivity (Hu et al.'s metric), (b) downstream cross-entropy degradation when SAE reconstructions are substituted (Engels et al.'s metric), and (c) circuit completeness scores (Chen et al.'s framework). This establishes the index's predictive validity for practical interpretability.

### Cross-Domain Insight

The key transplanted principle is from **rate-distortion theory** and the **information bottleneck method**. In lossy compression, an optimal encoder-decoder pair minimizes distortion at a given rate by being perfectly matched: the encoder discards information that the decoder does not need, and the decoder reconstructs only what the encoder preserves. When absorption occurs, this matching breaks: the encoder discards a feature (fails to activate the parent latent) while the decoder implicitly reconstructs it from another latent (the child). This is analogous to a communication channel where the transmitter and receiver have mismatched codebooks — the receiver can decode the message, but only by using the wrong codebook entry.

The structural correspondence holds because:
1. SAE training explicitly optimizes a rate-distortion tradeoff (sparsity = rate, reconstruction = distortion).
2. The encoder and decoder are jointly trained to be a matched pair.
3. Absorption is a suboptimal equilibrium where the matched pair property is violated.
4. Information-theoretic measures (mutual information, entropy) are the natural tools for quantifying matched-pair violations.

### Experimental Plan

**Experiment 1: Index Validation (Pilot: 15 minutes, Full: 45 minutes)**
- Load Gemma Scope SAEs for Gemma 2 2B (layers 0, 5, 10, 15) via SAELens.
- Compute Absorption Index for all latents on 10K tokens from OpenWebText.
- Compute ground-truth absorption rate via Chanin et al.'s method (first-letter classification).
- Measure Pearson correlation. If r < 0.5, iterate on the MI estimation method.

**Experiment 2: Deep-Layer Extrapolation (30 minutes)**
- Apply validated index to layers 18-26 of Gemma 2 2B.
- Plot absorption rate vs. layer depth.
- Compare with proxy metrics (feature frequency, decoder norm) to check for confounds.

**Experiment 3: Cross-Architecture Comparison (30 minutes)**
- Load multiple SAE architectures for the same Gemma 2 2B layer (TopK, JumpReLU, ReLU) from Gemma Scope / SAEBench.
- Compute Absorption Index for each.
- Compare distributions and rank architectures by absorption.

**Experiment 4: Downstream Impact Correlation (30 minutes)**
- On a subset of features, measure: (a) sensitivity via Hu et al.'s method (GPT-4.1-generated similar texts), (b) CE degradation from SAE reconstruction substitution, (c) circuit completeness via automated patching.
- Correlate each with Absorption Index.

**Experiment 5: GPT-2 Replication (20 minutes)**
- Replicate validation on GPT-2 small SAEs (SAELens pretrained) to test generalization across models.

### Resource Estimate

| Component | Cost | Notes |
|-----------|------|-------|
| Gemma 2 2B inference | Minimal | SAE analysis only, no model training |
| GPT-4.1 API for sensitivity | ~$5 | Only for Experiment 4 subset (~100 features) |
| Compute | 1x A100 / RTX 4090 | All experiments fit in <2 hours total |
| Storage | <10GB | SAE weights + activation caches |

All experiments are training-free, using existing pretrained SAEs. Total time: ~2.5 hours of compute, well within the project's constraints.

### Risk Assessment

**Risk 1: Low correlation with ground-truth absorption rate**
- **Mitigation**: The MI formulation is a starting point. If correlation is low, explore alternatives: (a) cosine similarity between encoder and decoder directions, (b) reconstruction error conditional on encoder activation, (c) decoder contribution variance when encoder is inactive. The framework is flexible.
- **Severity**: Medium. Would require iteration but not project abandonment.

**Risk 2: MI estimation noise on sparse activations**
- **Mitigation**: SAE activations are highly sparse (L0 ~ 20-100), making MI estimation challenging. Use adaptive binning or k-NN estimators designed for sparse data. Aggregate across multiple tokens to reduce variance.
- **Severity**: Low. Well-studied problem in information theory with established solutions.

**Risk 3: The metric conflates absorption with other phenomena (dead latents, feature splitting)**
- **Mitigation**: Include control experiments. Dead latents have near-zero encoder activation and near-zero decoder norm — easily filtered. Feature splitting (one true feature represented by multiple latents) has different encoder-decoder patterns (multiple encoder activations for one decoder direction). The index should be combined with existing metrics (L0, explained variance) to disambiguate.
- **Severity**: Medium. Requires careful experimental design but is addressable.

**Risk 4: Deep-layer results are unverifiable**
- **Mitigation**: This is inherent to the problem, not a project flaw. The value proposition is exactly that the index provides *the best available* absorption estimate for deep layers, validated by its strong correlation in shallow layers. Report confidence intervals based on the shallow-layer validation. Acknowledge the limitation explicitly.
- **Severity**: Low. The field already accepts this tradeoff for other deep-layer metrics.

### Novelty Claim

What is new:

1. **First quantitative, training-free absorption metric**: Prior work (Chanin et al.) noted encoder-decoder asymmetry as a "future work" direction but never developed it. The "Decoder-Free SAE" paper (January 2026) proposes eliminating the decoder rather than exploiting the asymmetry.

2. **First absorption measurement across all layers**: Current methods are limited to layers 0-17. This work enables the first systematic absorption analysis from layer 0 to the final layer.

3. **Information-theoretic grounding**: Prior absorption metrics are heuristic (ablation-based). This metric is grounded in rate-distortion theory and the information bottleneck, providing theoretical justification rather than empirical correlation alone.

4. **Cross-architecture comparison at depth**: Enables fair comparison of absorption-mitigating architectures (Matryoshka, HSAE, OrtSAE, JumpReLU) on deep layers for the first time.

5. **Predictive validity for downstream impact**: Connects the absorption metric to downstream interpretability outcomes (sensitivity, CE degradation, circuit completeness), establishing its practical value beyond intrinsic SAE quality.

What is NOT new (and properly cited):
- The observation of encoder-decoder asymmetry (Chanin et al., 2024/2025; Decoder-Free SAE, 2026)
- The rate-distortion framework for SAEs (Bereska et al., 2025)
- The ablation-based ground-truth absorption metric (Chanin et al., 2024/2025)
- The feature sensitivity metric (Hu et al., 2025)
- The downstream CE degradation analysis (Engels et al., 2024/2025)

---

*Sources:*
- [Chanin et al., A is for Absorption](https://arxiv.org/abs/2409.14507)
- [Hu et al., Measuring Sparse Autoencoder Feature Sensitivity](https://arxiv.org/abs/2509.23717)
- [Bereska et al., Superposition as Lossy Compression](https://arxiv.org/abs/2512.13568)
- [Engels et al., Decomposing The Dark Matter of Sparse Autoencoders](https://arxiv.org/abs/2410.14670)
- [Smith et al., Negative Results for SAEs](https://deepmindsafetyresearch.medium.com/negative-results-for-sparse-autoencoders-on-downstream-tasks-and-deprioritising-sae-research-6cadcfc125b)
- [Karvonen et al., SAEBench](https://arxiv.org/abs/2503.09532)
- [Bussmann et al., Matryoshka SAE](https://arxiv.org/abs/2503.17547)
- [Luo et al., HSAE](https://arxiv.org/abs/2602.11881)
- [Wu et al., MI-based SAE Interpretability](https://arxiv.org/abs/2502.15576)
- [Han et al., Causal Interpretation of SAE Features](https://arxiv.org/abs/2509.00749)
- [Marks et al., Unified Theory of Sparse Dictionary Learning](https://arxiv.org/abs/2512.05534)
- [Chen et al., Rethinking Circuit Completeness](https://arxiv.org/abs/2505.10039)
- [RouteSAE, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.346.pdf)
- [Decoder-Free SAE](https://arxiv.org/abs/2601.06478)
