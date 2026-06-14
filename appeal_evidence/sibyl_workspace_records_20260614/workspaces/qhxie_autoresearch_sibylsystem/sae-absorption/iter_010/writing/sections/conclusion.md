# Conclusion

This paper presents the first systematic cross-domain characterization of feature absorption in sparse autoencoders, extending measurement from the first-letter spelling benchmark to entity-attribute knowledge hierarchies on Gemma 2 2B. Five contributions emerge from the empirical program.

**1. Cross-domain absorption characterization with quantitative confound decomposition.** Absorption rates at layer 24 span a 4.1$\times$ descriptive range across four hierarchy types: 11.6% (city-language), 27.1% (first-letter), 31.4% (city-continent), and 45.1% (city-country). Within-RAVEL variation is statistically significant (Kruskal-Wallis $p = 7.4 \times 10^{-66}$). A probe degradation ablation---degrading first-letter probes via weight noise injection to seven $F_1$ levels from 0.69 to 1.0---yields a well-fitted absorption-vs-probe-quality curve ($R^2 = 0.777$, $p = 0.009$, Spearman $\rho = -1.0$). City-continent absorption matches the curve within 0.6 pp (fully explained by probe quality). City-language sits 21.3 pp below the curve, identifying it as a genuine hierarchy-specific outlier that probe quality alone cannot account for.

**2. Universal causal mechanism via activation patching.** Zeroing the identified child feature recovers parent probe predictions across all tested hierarchy types: first-letter ($d = 1.33$, $p = 0.000218$), city-continent ($d = 1.50$, $p < 10^{-20}$), and city-language ($d = 0.75$, $p < 10^{-18}$). Recovery rates range from 32.5% to 61.9% versus 1.5--6.8% for activation-magnitude-matched controls. Competitive exclusion is a universal absorption mechanism, not a first-letter-specific artifact.

**3. Decoder information entanglement.** Child decoder vectors carry 6.16 nats (first-letter, $N = 158$) and 3.98 nats (city-continent, $N = 1{,}464$) of parent-direction information, versus 0.012 nats for random-direction controls. Both hierarchies show 100% of instances exceeding all classification thresholds (0.05, 0.1, 0.2 nats). This diagnostic shares the probe direction with the false-negative classification; it measures decoder geometry, not computational redundancy.

**4. Hierarchy dominates architecture.** No significant architecture effect on absorption was detected (ANOVA $p = 0.50$--$0.53$ across JumpReLU, BatchTopK, and Matryoshka SAEs). Hierarchy type is the sole significant predictor ($p = 0.005$--$0.041$). This comparison is underpowered (12 observations) and should be interpreted as "effect not detected," not "no effect exists."

**5. Quadruple negative for correlational predictors.** GAS ($\rho = 0.116$, AUROC $= 0.571$), CMI ($\rho = 0.044$, $p = 0.84$), the Absorption Tax ($\rho = -0.20$, concordance at chance), and a rate-distortion three-factor model ($R^2 = 0.104$, all individual predictors in the wrong direction) all fail to predict absorption. This establishes a methodological boundary: absorption is a causal phenomenon driven by encoder competitive dynamics, not by static geometric or information-theoretic properties of the trained decoder.

## Limitations

All experiments use a single model (Gemma 2 2B). Generalization to other architectures (Llama, Pythia), model scales, and training paradigms (instruction-tuned, RLHF) is untested. The probe degradation ablation resolves the direction of the confound ($R^2 = 0.777$) but the $F_1 = 1.0$ control absorption rate (21.6%) falls below the iter\_009 baseline CI $[26.3\%, 34.7\%]$ due to per-token versus per-word aggregation differences. Token position asymmetry (first-letter at position $-6$, RAVEL at position $-2$) is uncontrolled in cross-framework comparisons. The decoder entanglement diagnostic has acknowledged circularity. The architecture comparison is underpowered. City-country probe quality ($F_1 = 0.73$) falls below both quality gates; its 45.1% absorption rate should be treated as exploratory.

## Future Work

Three directions follow from these results. First, higher-quality probes for entity-attribute hierarchies---via contrastive learning, richer prompt templates, or embedding-based classifiers---would reduce the probe quality confound and sharpen the cross-domain signal. Second, extending the measurement to larger models (Gemma 2 9B, Llama 3) and structurally diverse hierarchies (taxonomic, part-of-speech, factual person-profession) would test whether the city-language suppression effect and the universal competitive exclusion mechanism generalize. Third, a genuine computational-redundancy test---activation-level ablation ($z_{\text{parent}} = 0$) or path patching through independent circuits---would determine whether absorbed parent information is computationally recoverable by the model, resolving the question that the decoder entanglement diagnostic cannot answer due to its circularity.

The probe degradation ablation is the methodological contribution with the broadest applicability. Any future cross-domain absorption study that does not include a probe quality control risks conflating measurement artifacts with genuine hierarchy effects. The first-letter task, with its $F_1 = 1.0$ probes, remains the only hierarchy where the absorption rate is free of this confound---a reason to continue using it as a calibration anchor, not to rely on it as the sole benchmark.

<!-- FIGURES
- None
-->
