# 6. Discussion

## 6.1 The Inhibition Graph as a Mechanistic Explanation

The LCA--SAE structural correspondence is exact, not metaphorical. For tied-weight SAEs, $G = W_{\text{dec}}^T W_{\text{dec}}$ is exactly the LCA inhibition matrix. For untied SAEs---the standard case---decoder correlations encode competitive relationships because decoder directions that reconstruct the same input variance must compete for activation. This correspondence provides a causal mechanism for absorption: competitive suppression.

The mechanism resolves an apparent paradox from our prior experiments. Absorption is real (detectable via differential correlation) yet its downstream consequences are limited (steering remains effective, probing is unaffected). Competitive suppression explains this by distinguishing two components of a feature: its decoder direction $W_{\text{dec}}[i]$ (preserved, enabling steering) and its encoder activation $z_i$ (suppressed, causing recall loss). Steering operates on decoder directions, so it is robust to encoder suppression. Probing at high $k$ uses many latents, so the loss of one parent's activation is compensated by others. Only when we isolate the feature-specific contribution (via delta correction) does the suppression effect become detectable.

The precision--recall asymmetry---precision invariant near 1.0 while recall varies from 0.10 to 1.00---is the signature prediction of competitive suppression. Inhibition suppresses true positives (parent fails to fire when child fires) but does not cause false positives (parent does not fire for incorrect inputs). This selectivity-preserving, coverage-reducing effect is exactly what the LCA framework predicts.

## 6.2 Why Prior Work Found Null Results

Our prior experiments (H1--H5) produced predominantly null correlations between absorption and downstream task performance. The inhibition framework explains why.

**Raw steering metrics confound absorption-specific effects with generic directional bias.** Random feature steering achieves 34--38% success, demonstrating that arbitrary decoder directions produce non-negligible steering effects. This generic bias masks the feature-specific contribution that competitive suppression degrades. Delta-corrected steering subtracts this baseline, isolating the unique information lost to inhibition. The significant H1b result at layer 8 ($r = -0.431$, $p = 0.028$) emerges only after this correction because it measures the incremental degradation beyond generic directional effects.

**Low absorption variance compresses effect sizes.** The distribution of absorption rates is strongly right-skewed: 18--22 of 26 features per layer show absorption below 10%. With $n = 26$ features, statistical power to detect a Pearson correlation of $|r| \geq 0.50$ at $\alpha = 0.05$ is approximately 65%. The H6--H10 experiments address this limitation by changing the prediction target: rather than correlating absorption with task performance (a noisy, indirect relationship), we test whether decoder correlations directly predict absorption pairs---a structural prediction with a clear chance baseline.

**Layer-dependent effects reflect depth-varying inhibition strength.** The significant H1b result occurs only at layer 8, not at layer 4. The inhibition framework predicts this: deeper layers have stronger hierarchical structure, producing stronger competitive dynamics and thus more detectable suppression effects. H9 tests this prediction directly by measuring whether mean edge weight increases with layer depth.

## 6.3 Practical Implications

The inhibition framework yields four practical contributions for SAE researchers and practitioners.

**Diagnostic tool.** Practitioners can identify at-risk features without running absorption metrics. Computing total incoming inhibition $\text{inh}_{\text{in}}(i) = \sum_{j \in N(i)} |G_{ji}|$ for a latent takes milliseconds and requires only pretrained weights. Features with high incoming inhibition are more likely to be absorbed, enabling pre-emptive feature selection for downstream tasks.

**Training-free repair.** Homeostatic rebalancing restores parent firing with a single-pass correction on pretrained SAEs. Unlike Matryoshka SAEs, OrtSAE, and HSAE---all of which require retraining---homeostatic rebalancing operates on existing SAE weights. The reconstruction constraint ($\Delta_{\text{recon}} < 5\%$) ensures that repair does not degrade the SAE's primary function.

**Layer selection guidance.** Deeper layers exhibit stronger competitive dynamics. When choosing which layer to analyze for a given feature, practitioners should expect stronger absorption effects (and thus stronger inhibition structure) in mid-to-late layers. Layer 8 in GPT-2 Small shows the most pronounced effects, consistent with the H1b significance at that layer.

**Metric design.** The framework justifies delta-corrected metrics as essential for steering evaluation. Raw steering metrics conflate feature-specific contribution with generic directional bias; baseline subtraction isolates the incremental effect that competitive suppression degrades. The field should adopt delta-corrected steering as standard practice.

## 6.4 Relationship to Existing Solutions

The inhibition framework provides theoretical grounding for existing architectural solutions and clarifies their mechanisms.

Matryoshka SAEs (Bussmann et al., 2025) reduce absorption by encoding broader concepts at coarser granularities. The inhibition framework explains why this helps: by separating parent and child features into different dictionary levels, Matryoshka SAEs reduce the decoder correlations $G_{ij}$ between hierarchically related features, weakening competitive suppression.

OrtSAE (Korznikov et al., 2025) enforces decoder orthogonality, reducing absorption by 65%. The inhibition framework explains this too: orthogonal decoders have $G_{ij} = \langle W_{\text{dec}}[i], W_{\text{dec}}[j] \rangle \approx 0$ for $i \neq j$, eliminating competitive suppression entirely. The cost is slightly lower explained variance, because orthogonal directions cannot pack as many features into the same model dimension.

HSAE (Luo et al., 2026) explicitly models hierarchical structure. The inhibition framework provides a theoretical justification: hierarchical structure predicts which feature pairs will have high $G_{ij}$ and thus which pairs will exhibit competitive suppression. HSAE's structural constraints target exactly these high-correlation pairs.

Our approach complements these solutions: while they modify architecture or retrain, we provide a training-free diagnostic and repair that works on any pretrained SAE.

## 6.5 Limitations of the Inhibition Framework

The framework has several limitations that bound its applicability.

**Local graph scope.** The top-k neighbor restriction may miss long-range absorption relationships where a parent is suppressed by a distant latent with weak but consistent correlation. We test multiple $k$ values (10, 20, 50) to assess this, but a global analysis of the full correlation matrix may reveal additional structure.

**Feature set scope.** Our validation uses first-letter features (A--Z), which have a shallow, uniform hierarchy. Semantic hierarchies (e.g., "animal" $\rightarrow$ "mammal" $\rightarrow$ "dog") have deeper, more asymmetric structure that may produce different inhibition patterns. Cross-domain validation is needed.

**Single model family.** All experiments use GPT-2 Small (124M parameters). Larger models (Gemma-2-2B, Llama-3.1-8B) have deeper hierarchies, larger dictionaries, and potentially stronger absorption. The inhibition framework predicts stronger effects in larger models, but this remains to be tested.

**Tied-weight assumption.** The exact structural correspondence $G = W_{\text{dec}}^T W_{\text{dec}}$ assumes tied encoder-decoder weights. Standard trained SAEs use untied weights, for which the correspondence is approximate. We argue that decoder correlations still encode competitive relationships because decoder directions must compete to explain input variance, but the approximation quality varies across SAE architectures.

**Homeostatic rebalancing is single-pass.** Our repair applies a single forward-pass correction. Iterative optimization---adjusting $\alpha$ per-feature, or running multiple rebalancing steps---may yield better results but would require more computation and careful convergence analysis.

<!-- FIGURES
- None
-->
