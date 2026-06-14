# 8. Conclusion

This study provides the first systematic cross-domain characterization of feature absorption in sparse autoencoders, extending measurement from a single proxy task (first-letter spelling on GPT-2 Small) to entity-attribute knowledge hierarchies on Gemma 2 2B. Five contributions emerge from the analysis.

**1. Cross-domain absorption characterization.** Absorption rates vary 4$\times$ across hierarchy types at layer 24: from 11.6% (city-language) to 45.1% (city-country) on 16k SAEs (Kruskal-Wallis $H$ = 299.95, $p$ = 7.4 $\times 10^{-66}$, $N$ = 3,566). No simple semantic-versus-syntactic ordering holds -- city-country (45.1%) exceeds first-letter (27.1%), while city-language (11.6%) falls far below it. The pattern is hierarchy-specific: absorption severity depends on the structure of the parent-child relationship, not on whether the hierarchy is syntactic or semantic. Absorption concentrates at the final prediction layer, rising 18$\times$ from layer 6 (2.4%) to layer 24 (42.9%) for first-letter, implicating task-specific computation rather than generic representation failure.

**2. Causal evidence for competitive exclusion.** Activation patching on first-letter spelling confirms that zeroing a single child feature recovers the parent probe's prediction in 32.5% of false negative cases (control: 1.5%, Wilcoxon signed-rank $p$ = 0.000218, Cohen's $d$ = 1.33, $n$ = 25 words). This is the first interventional -- not merely correlational -- evidence for competitive exclusion in SAEs. The same intervention fails for city-continent (recovery: 0.05%, control: 14.5%, $d$ = $-$0.91), revealing a mechanistic divide between concentrated (single-feature) and distributed (multi-feature) absorption.

**3. Absorption is always pathological.** Across 1,471 false negative instances from 50 city-continent entities, 0% qualify as benign at any threshold tested ($\tau$ = 0.05, 0.1, 0.2). Ablating the parent direction from child decoder vectors produces a mean $|\Delta_{\text{logit}}|$ = 3.98 nats -- approximately 1,000$\times$ the control perturbation (0.004 nats). The hypothesis that absorption faithfully reflects computational redundancy is decisively falsified.

**4. Hierarchy dominates architecture.** SAE architecture has no significant effect on absorption rates (Kruskal-Wallis $p$ = 0.50 at L24, $p$ = 0.75 at L12 across JumpReLU, BatchTopK, and Matryoshka). Hierarchy type is the sole significant predictor ($p$ = 0.005 at L12). Switching SAE architectures will not resolve absorption.

**5. Comprehensive negative results.** Five correlational approaches fail to predict absorption: the Geometric Absorption Score ($\rho$ = 0.116), conditional mutual information ($\rho$ = 0.044), the Absorption Tax ranking ($\rho$ = $-$0.20), rate-distortion predictors ($R^2$ = 0.088 with individual predictors in the opposite direction to hypothesized), and competition coefficients (non-significant). These failures, each reported with full metrics and confidence intervals, delineate a clear boundary: absorption resists prediction from readily computable feature statistics.

### Limitations

Four limitations bound the conclusions. First, all experiments use a single base model (Gemma 2 2B); generalization to other architectures (Llama, Mistral) and larger scales (9B, 27B) is untested. Second, RAVEL probes achieve $F_1$ = 0.73--0.87 for entity-attribute hierarchies, below the strict quality gate ($F_1 \geq$ 0.90); measured absorption rates for these hierarchies are upper bounds. The city-country rate (45.1%) should be treated with particular caution given its $F_1$ = 0.73. Third, the causal evidence for competitive exclusion is limited to first-letter spelling; the cross-domain activation patching result (reversed direction) constrains causal claims to the concentrated absorption regime. Fourth, the architecture comparison is limited by width mismatch (Matryoshka 32k vs. others at 16k/65k) and probe quality at layer 12.

### Future Directions

The concentrated-versus-distributed mechanistic divide motivates three research directions. First, multi-feature distributed absorption detection methods -- which must account for collective effects rather than single parent-child pairs -- are needed to characterize absorption in semantic hierarchies where no single child feature is responsible. Second, better probes for entity-attribute hierarchies (contrastive learning, richer prompt templates, larger entity sets) would tighten cross-domain absorption estimates and clarify whether the 4$\times$ variation reflects genuine differences or measurement noise. Third, extending the cross-domain framework to larger models (Gemma 2 9B, Llama 3) and additional hierarchy types (temporal, causal, taxonomic) would test whether the hierarchy-specific patterns observed here generalize across model families and knowledge domains.

The consistent failure of correlational approaches (Section 7.1) suggests a broader methodological lesson: unsupervised absorption detection must account for encoder dynamics, not just decoder geometry. Future work should explore circuit-level analyses that trace how the SAE encoder arbitrates between competing features during the encoding step -- the mechanistic locus where absorption originates.

<!-- FIGURES
- None
-->
