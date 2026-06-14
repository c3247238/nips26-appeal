# 6. Discussion

## 6.1 Why the Mixed Result?

We consider four explanations for the pattern of results: null correlations for H1 and H2, a significant negative correlation for H1b at layer 8, and failure of H3.

**Raw steering is confounded by baseline effects.** Random feature steering achieves 34--38% success on our task, demonstrating that arbitrary decoder directions produce non-negligible steering effects. This generic directional bias masks the feature-specific contribution. When we subtract the random baseline, the true relationship between absorption and steering effectiveness emerges at layer 8. The implication is clear: raw steering metrics without baseline controls may be misleading. A feature that achieves 80% raw steering success may appear effective, but if random directions achieve 38%, the feature-specific contribution is only 42 percentage points---and this delta may correlate negatively with absorption.

**Low absorption variance compresses effect sizes.** The distribution of absorption rates is strongly right-skewed: 18--22 of 26 features per layer show absorption below 10%. The restricted range compresses any potential correlation. With $n = 26$ features, the statistical power to detect a Pearson correlation of $|r| \geq 0.50$ at $\alpha = 0.05$ is approximately 65%. The H1b correlation at layer 8 ($r = -0.431$) is close to this threshold and achieves significance, but H1 ($r = -0.301$) and H2 ($r = -0.107$) fall below it. A feature set with greater absorption variance---semantic hierarchies with deeper structure, or larger models with stronger absorption---might reveal stronger correlations.

**Layer-dependent effects suggest depth matters.** The significant H1b result occurs only at layer 8, not at layer 4. This layer specificity suggests that absorption's impact on steering may depend on where in the network the feature resides. Deeper layers (closer to output) may have more concentrated feature representations where absorption has stronger consequences, or the decoder directions at deeper layers may be more sensitive to activation redistribution. Alternatively, the effect may be a statistical fluctuation: with multiple comparisons across layers and hypotheses, one significant result at $\alpha = 0.05$ is expected by chance. We caution against over-interpreting the layer 8 finding without replication.

**Probing is inherently robust to absorption.** k-sparse probing F1 depends on whether a small set of latents can classify the target concept. If a parent feature is absorbed, child features or correlated latents may still activate on the same concept, allowing the probe to achieve high F1. Full-activation probing (using all 24,576 latents) consistently outperforms k-sparse probing, confirming that task-relevant information is distributed and recoverable. This distributed encoding may make probing inherently resilient to the loss of individual parent features.

## 6.2 Implications for the Field

Our findings carry several implications for SAE research and practice.

First, absorption as currently measured may degrade steering effectiveness when properly evaluated, but not probing. The field has devoted substantial effort to absorption reduction---Matryoshka SAEs, OrtSAE, and HSAE all prioritize this objective. Our H1b result suggests that this effort may be partially justified for steering applications, but only when delta-corrected metrics are used. For probing, practitioners can use absorbed features without concern for absorption-related degradation.

Second, the field should adopt delta-corrected steering as a standard evaluation practice. Raw steering metrics conflate feature-specific contribution with generic directional bias, producing misleading null results. Any steering evaluation should include a random feature baseline and report delta success. This is analogous to the use of control conditions in experimental psychology: without subtraction of baseline effects, true relationships are obscured.

Third, task-relevant evaluation should supplement structural metrics. SAEBench includes absorption alongside sparsity and reconstruction error, but none of these metrics have been validated against downstream interpretability tasks. A SAE with low absorption may still produce unreliable features for circuit analysis; conversely, a SAE with higher absorption may perform adequately for steering (when delta-corrected) and probing. Task-oriented benchmarks that measure steering fidelity, probing accuracy, and circuit recovery rate would provide more actionable guidance.

## 6.3 Comparison with Pilot

Our pilot experiment (layer 8, 50 samples per feature) yielded $r = -0.153$, $p = 0.456$ for H1 (raw steering). The full experiment (layer 8, 100 samples per feature) strengthened the negative trend to $r = -0.301$, $p = 0.136$ for H1 but did not achieve significance. However, the critical addition of the random baseline control revealed H1b significance ($r = -0.431$, $p = 0.028$) that was invisible in the raw data. Doubling the sample size reduced variance in steering success estimates, but the baseline subtraction was the decisive methodological improvement.

## 6.4 What Would Change Our Conclusion?

We identify four conditions under which our conclusion---that absorption's impact is mixed, layer-dependent, and only detectable with delta-corrected steering metrics---might not generalize.

**Larger models.** GPT-2 Small (124M parameters, 85M non-embedding) may not exhibit absorption as strongly as larger models. Gemma-2-2B and Llama-3.1-8B have deeper hierarchies and larger dictionaries, potentially producing higher absorption rates and stronger task degradation.

**Semantic hierarchy features.** First-letter features (A--Z) have a shallow, uniform hierarchy: each letter branches to words starting with that letter. Semantic hierarchies (e.g., "animal" $\rightarrow$ "mammal" $\rightarrow$ "dog" $\rightarrow$ "poodle") have deeper, more asymmetric structure that may produce stronger absorption and clearer task degradation.

**Alternative absorption metrics.** The Chanin differential correlation metric is one of several absorption measures. SAEBench includes an ablation-based metric that may capture different failure modes. JumpReLU SAEs reportedly show stronger absorption under alternative metrics. A different metric might reveal a relationship that differential correlation misses, or it might fail to replicate our H1b finding.

**Different downstream tasks.** Steering and probing are two of many interpretability applications. Circuit finding with activation patching and model editing require precise feature isolation and may be more sensitive to absorption. Tasks that depend on feature composition (e.g., tracing how "starts with A" and "is a fruit" combine to produce "Apple") may also show stronger degradation.

<!-- FIGURES
- None
-->
