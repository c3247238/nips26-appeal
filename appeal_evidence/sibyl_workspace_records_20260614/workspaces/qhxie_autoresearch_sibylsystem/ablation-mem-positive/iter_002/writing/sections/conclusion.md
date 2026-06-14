## 6. Conclusion

### 6.1 Summary

This work addresses three gaps in the feature absorption literature: the scalability of semantic probe datasets, the generalization of training-free detection across SAE architectures, and the stability of absorption metrics across model families. Our findings, derived from 60 semantic probes across two architectures (GemmaScope JumpReLU and GPT-2 ReLU), yield four concrete results.

First, expanding from 5 to 15 hyponyms per category eliminates failed probes entirely. All 30 probes on GemmaScope achieve AUROC $>$ 0.7, with mean AUROC = 0.980 $\pm$ 0.034. This validates the scaled probe pipeline as a reliable foundation for absorption measurement.

Second, projection absorption is stable across architectures. GemmaScope JumpReLU achieves 98.2% $\pm$ 1.2% and GPT-2 ReLU achieves 91.2% $\pm$ 5.2%---a 7.7% difference that falls well within our 10% stability threshold (H3: PASS). The Mann-Whitney U test confirms the difference is significant ($p < 0.001$) and Cohen's $d = 1.82$ indicates a large effect size by conventional thresholds, reflecting the tight variance on GemmaScope rather than a large practical gap. Both architectures show 100% of probes crossing the 0.5 projection threshold.

Third, H2 failed as originally stated. The hypothesis that $A_j$ correlation would be higher on GPT-2 ReLU ($\rho > 0.6$) due to unconstrained decoder norms was falsified: mean $\rho = -0.194$ across layers, far below the 0.6 threshold, and decoder norms were constrained rather than unconstrained. However, this failure revealed an exploratory finding: a layer-dependent $A_j$ correlation pattern that peaks at mid-layers. On GPT-2 layer 8 (relative depth 0.67), $A_j$ correlates with projection absorption at $\rho = 0.705$ ($p = 0.023$), while layers 5 and 10 show negative correlations ($\rho = -0.590$ and $-0.697$). The sign flip is statistically significant ($z = 3.253$, $p = 0.0011$ for layer 8 vs 10). With only 3 layers tested and correlations computed over $n = 10$ categories, this observation requires validation on additional layers and models before generalizing. If confirmed, the pattern establishes layer depth---not architecture---as the primary moderating factor for training-free detection.

Fourth, decoder norm constraints are consistent across the two architectures tested. Both GPT-2 ReLU (mean = $1.000045 \pm 5.4 \times 10^{-6}$) and GemmaScope JumpReLU maintain decoder norms approximately 1.0. This contradicts the hypothesis that unconstrained norms explain detector differences and suggests that decoder normalization may emerge from training dynamics rather than architectural design alone, though architectural effects cannot be ruled out with only two SAE families.

A secondary finding with practical implications: the ablation-based metric remains functionally insensitive across both architectures. Mean ablation scores are near-zero: 0.0016 on GemmaScope E3v2 and 0.0192 on GPT-2 E7. These near-zero scores confirm that functional insensitivity is not architecture-specific. The redundancy inherent in absorbed representations---where parent concepts are encoded across multiple latents---renders ablation-based detection unreliable as a standalone metric.

### 6.2 Future Work

Four directions emerge from our findings, each tied to a specific limitation.

**Additional model families.** Our analysis is limited to two decoder-only transformers (Gemma-2-2B and GPT-2 Small), leaving open whether the layer-dependent $A_j$ pattern generalizes. Extending to Pythia (Biderman et al., 2023), Llama (Touvron et al., 2023), and encoder-decoder architectures like T5 would test whether the layer-dependent $A_j$ pattern generalizes across model families.

**Full layer correlation landscape.** With only three layers tested per model, we cannot determine whether the mid-layer peak at relative depth $\sim$0.67 is a fixed property or scales with model depth. Mapping layers 3, 6, 7, 9, and 11 on GPT-2 would test whether the peak is localized or part of a broader pattern. Mapping a representative subset of Gemma-2-2B's 26 layers would clarify whether the peak occurs at a fixed relative depth (0.6-0.7) or scales with model depth.

**Feature-level analysis of the mid-layer peak.** The sign flip between layers 8 (positive) and 10 (negative) suggests that $A_j$ captures different phenomena at different depths. Activation patching on individual latents, causal ablation studies, and layer-wise feature clustering would distinguish three hypotheses: (i) mid-layers have the most pronounced feature hierarchies; (ii) deep layers encode distributed, context-dependent features that confound the detector; (iii) shallow layers lack sufficient semantic structure for discrimination.

**Layer-aware training-free detector.** The current $A_j$ detector uses a single global threshold. Our results suggest that layer-specific thresholds---calibrated to the observed correlation pattern---would improve detection accuracy. A detector that applies positive $A_j$ weighting at mid-layers and negative weighting at shallow and deep layers could achieve higher overall correlation than any single-layer detector.

### 6.3 Broader Impact

Our findings carry two implications for the mechanistic interpretability community. First, projection-based absorption metrics should replace ablation-based metrics as the default in SAE evaluation benchmarks. The gap between ablation and projection rates (0-33% versus 91-98%) means that benchmarks reporting only ablation-based absorption systematically undercount absorbed features. SAEBench (Karvonen et al., 2025) and CE-Bench (Peng et al., 2025) would benefit from incorporating projection-based rates alongside existing metrics.

Second, training-free detectors require layer-aware calibration. A single $A_j$ correlation averaged across layers yields $\rho = -0.194$ on GPT-2, masking the strong mid-layer signal. Stratifying by layer depth reveals that the detector is highly effective at mid-layers ($\rho = 0.705$) and ineffective elsewhere. Future benchmarks should report training-free metrics stratified by relative layer depth, with layer-specific thresholds replacing global ones.

The architecture stability of projection absorption---7.7% difference across JumpReLU and ReLU---provides a foundation for cross-model comparison that has been missing from the literature. As the community develops new SAE architectures (Matryoshka SAEs, OrtSAE, MP-SAE), projection-based metrics offer a consistent baseline for measuring whether architectural innovations genuinely reduce absorption or merely shift its manifestation.

<!-- FIGURES
- None
-->
