## 5. Discussion

### 5.1 Architecture Stability of Projection Metrics

Why do JumpReLU and ReLU SAEs---which differ in activation thresholding, dictionary size (by a factor of 1.5), and underlying model scale (2B vs 124M parameters)---produce projection absorption rates that differ by only 7.7 percentage points? The small absolute gap suggests that projection-based absorption captures a structural property of sparse autoencoding that is largely independent of architectural specifics. Both the non-zero threshold in JumpReLU and the standard ReLU activation converge to representations where a single latent dominates the probe weight vector, even if the precise fraction varies.

The higher variance on GPT-2 (std = 0.052 vs 0.012 on GemmaScope) is the more informative difference. It suggests that ReLU SAEs produce more heterogeneous absorption patterns across semantic categories. The animal probe at GPT-2 layer 8, for instance, shows $A_{\text{proj}} = 0.770$---the lowest value across all 60 probes---while the same category on GemmaScope layer 5 achieves 0.975. Whether this heterogeneity reflects genuine architectural differences (e.g., the absence of thresholding in ReLU SAEs allows more distributed encoding for certain concepts) or sampling variation within categories requires investigation with more probes per layer.

The consistently high projection absorption rates ($>$90%) on both standard architectures raises a question for the architectural solutions literature: if OrtSAE reduces absorption by 65% relative to standard ReLU SAEs (Korznikov et al., 2025), what metric does it improve? Our probe-based projection metric may capture a different facet of absorption than the metrics used to evaluate OrtSAE or MP-SAE. Cross-benchmark validation---measuring the same SAEs with both SAEBench's metric and our probe-based metric---would clarify whether architectural innovations reduce all forms of absorption or only specific manifestations.

### 5.2 Decoder Norm Constraints Across Architectures

The finding that GPT-2 ReLU SAEs maintain decoder norms of approximately 1.0---despite having no explicit architectural constraint---suggests that decoder normalization emerges from training dynamics rather than design. SAELens trains GPT-2 SAEs with gradient descent on reconstruction loss plus an L1 sparsity penalty; GemmaScope trains JumpReLU SAEs with a similar objective but adds the JumpReLU threshold parameter. Both procedures converge to column norms tightly clustered around 1.0, which may reflect an implicit bias in the optimization landscape: unit-norm decoder vectors may provide the most favorable sparsity-reconstruction trade-off.

This has a direct implication for interpreting the $A_j$ detector. If decoder norms are universally constrained, the numerator $\|d_j\|^2$ becomes approximately constant across latents, and $A_j$ reduces to a proxy for encoder alignment $(d_j^\top e_j)^{-1}$. The layer-dependent correlation pattern we observe therefore reflects layer-varying encoder-decoder alignment structures rather than norm-driven absorption differences. This reframes $A_j$ from a norm-based detector to an alignment-based detector, with alignment itself varying systematically by network depth.

### 5.3 Layer-Dependent Detection Pattern

The non-monotonic $A_j$ correlation pattern---positive at mid-layers, negative at shallow and deep layers---reframes how practitioners should use training-free detectors. Rather than applying a single global threshold across all layers, $A_j$ should be treated as a layer-conditional screening tool: reliable at relative depths of 0.6-0.7 but potentially misleading elsewhere.

Three mechanisms could explain the mid-layer peak. First, feature hierarchies may be most pronounced at intermediate depths, where abstract concepts have been composed from lower-level features but not yet compressed into distributed representations for output prediction. At layer 8 of GPT-2 (relative depth 0.67), the model has processed sufficient context to form category-level representations while still maintaining relatively sparse feature codes. The $A_j$ detector, which assumes a direct relationship between decoder norm and absorption, may align well with the geometry of these mid-layer representations.

Second, deep layers (layer 10, relative depth 0.83) may encode more distributed, context-dependent features that are less amenable to single-latent absorption detection. The negative correlation at layer 10 suggests that high $A_j$ values coincide with low projection absorption---the opposite of the hypothesized relationship. One interpretation is that deep-layer latents with high $A_j$ are not absorbed features but rather polysemantic features that activate across multiple unrelated concepts, a pattern related to feature hedging (Chanin et al., 2025a).

Third, shallow layers (layer 5, relative depth 0.42) may lack sufficient semantic structure for the $A_j$ detector to discriminate absorbed from non-absorbed features. The negative correlation at layer 5 ($\rho = -0.590$, $p = 0.073$, marginal) does not reach significance, possibly reflecting a genuine absence of hierarchical feature structure at shallow depths where representations are dominated by low-level syntactic and lexical features rather than semantic categories.

For practitioners, the implication is clear: a single $A_j$ correlation averaged across layers yields $\rho = -0.194$ on GPT-2, masking the strong mid-layer signal. Screening tools should stratify by relative layer depth, applying positive $A_j$ weighting at mid-layers and treating shallow and deep layers as uninformative for absorption detection.

### 5.4 Ablation Metric Insensitivity Is Universal

While the ablation metric flags 0-33% of probes as absorbed (0.0% on GemmaScope E3v2, 33.3% on GPT-2 E7), the projection metric flags 91-98%. This 60-70 point gap---not the low ablation rate itself---is the evidence for functional insensitivity. Mean ablation scores are near-zero: $0.0016 \pm 0.0082$ on GemmaScope and $0.0192 \pm 0.0358$ on GPT-2. This functional insensitivity is not architecture-specific; it appears on both JumpReLU and ReLU SAEs.

The insensitivity has a straightforward explanation. When a feature is absorbed, the parent concept is redundantly encoded across multiple latents. Ablating the top latent leaves sufficient residual signal in other latents for the probe to maintain accuracy. The ablation score $A_{\text{abl}} = \text{acc}_{\text{full}} - \text{acc}_{\text{ablated}}$ therefore remains near zero even when projection absorption $A_{\text{proj}}$ is high. This is not a bug in the metric but a fundamental property of absorbed representations: by definition, absorption implies redundancy.

The practical implication is that ablation-based metrics should not be used as the sole criterion for absorption detection. Projection-based metrics are more sensitive and correlate with the underlying geometry of probe weights. For training-free detection, the $A_j$ metric shows promise at specific layers but requires layer-aware calibration to account for the non-monotonic correlation pattern.

### 5.5 Limitations

Our study has four principal limitations. First, we evaluate only two model families (Gemma-2-2B and GPT-2 Small). Both are decoder-only transformers, and our findings may not generalize to encoder-decoder architectures, mixture-of-experts models, or state-space models. Second, we sample only three layers per model. The layer-dependent $A_j$ correlation pattern is based on three points (layers 5, 8, 10), and the non-monotonic shape---while statistically significant---could change with denser sampling. A full layer correlation landscape would require testing all 12 GPT-2 layers and a representative subset of Gemma-2-2B's 26 layers.

Third, our semantic probes are limited to 10 WordNet categories of concrete nouns. Abstract concepts (e.g., emotions, spatial relations), verbs, and adjectives may exhibit different absorption patterns. The WordNet hierarchy provides a clean experimental framework, but real-world language involves richer semantic structures that may challenge the probe-based detection pipeline.

Fourth, the $A_j$ correlation pattern needs validation on additional models. Our finding of a mid-layer peak at relative depth 0.67 may be specific to GPT-2's 12-layer architecture. Gemma-2-2B has 26 layers, and the corresponding relative depth would be layer 17 or 18---depths we did not test. Whether the peak is at a fixed relative depth or scales with model depth remains an open question.

### 5.6 Implications for SAE Evaluation Benchmarks

Our findings have direct implications for SAEBench (Karvonen et al., 2025) and other SAE evaluation frameworks. Current benchmarks report absorption as a single scalar metric, but our results show that absorption is layer-dependent and metric-dependent. A benchmark that reports only ablation-based absorption would miss 60-70% of absorbed features (comparing 30-33% ablation rate to 91-98% projection rate). A benchmark that reports only $A_j$ without layer stratification would produce misleading results: on GPT-2, a single $A_j$ correlation averaged across layers would yield $\rho = -0.194$, suggesting the detector is not uniformly valid across layers, when in fact layer 8 shows strong positive correlation.

**Concrete recommendations for benchmark design.** We propose four specific changes to existing benchmark protocols.

First, absorption reporting should use a two-tier structure. Tier 1 reports the overall projection absorption rate across all tested layers (our primary cross-architecture metric). Tier 2 reports per-layer stratification: for each sampled layer, report projection absorption mean, standard deviation, and the fraction of probes exceeding the 0.5 threshold. This structure allows practitioners to compare overall rates across models while detecting layer-specific anomalies.

Second, ablation-based metrics should be reported as a sensitivity check rather than a primary measure. Benchmarks should report the ablation rate alongside the projection rate, with the gap between them (60-70 percentage points in our data) flagged as an indicator of metric reliability. A gap exceeding 50 points suggests the ablation metric is functionally insensitive for that model, and projection-based rates should be treated as the authoritative measure.

Third, training-free detectors require layer-aware thresholds. Rather than a single global $A_j$ threshold, benchmarks should define three zones based on relative layer depth: shallow layers (relative depth $<$ 0.5) where $A_j$ is uninformative, mid-layers (relative depth 0.5-0.8) where positive $A_j$ values indicate absorption, and deep layers (relative depth $>$ 0.8) where $A_j$ may be anti-correlated with absorption. A benchmark could report the $A_j$--projection correlation per layer and flag architectures where the correlation pattern deviates from the expected mid-layer peak.

Fourth, decoder norm constraints should be verified and reported as a model property. Since $A_j$ interpretation depends on whether decoder norms are constrained, benchmarks should include a norm check: compute the mean and standard deviation of $\|d_j\|$ across all latents and report whether norms are tightly clustered (standard deviation $<$ 0.01, as in both our architectures) or widely dispersed. This norm report enables correct interpretation of $A_j$ values: when norms are constrained, $A_j$ reflects encoder-decoder alignment rather than norm-driven absorption differences.

These recommendations require minimal additional computation. Projection absorption uses the same probe weights already computed by SAEBench's sparse probing metric. The $A_j$ detector requires only a single matrix operation over SAE weights. Decoder norm statistics are a byproduct of $A_j$ computation. The primary cost is stratified reporting---storing per-layer results rather than a single scalar---which is negligible compared to the cost of probe training or model inference.

<!-- FIGURES
- None
-->
