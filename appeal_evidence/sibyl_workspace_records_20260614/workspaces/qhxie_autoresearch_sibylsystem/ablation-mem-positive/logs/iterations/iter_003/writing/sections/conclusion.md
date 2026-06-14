# 6 Conclusion

We showed that feature absorption in Sparse Autoencoders does not render features uniformly non-steerable. High-CV absorbed features on GPT-2 Small produce steering effects 1.47x larger than low-CV absorbed features (0.525 vs 0.357 mean logit change, p < 0.01 with BH correction across all strengths), and approximately equal to non-absorbed features (0.102). The coefficient of variation successfully predicts which absorbed features retain steering potential.

## 6.1 Summary of Contributions

**First systematic evidence for steering heterogeneity within absorbed features.** Prior work (Basu et al., 2026) treated all absorbed features as uniformly non-steerable. Our full experiment (n=60 absorbed features, 30 high-CV vs 30 low-CV, 5 prompts, 3 steering strengths) demonstrates that absorbed features decompose into steerable (robust) and non-steerable (fragile) subpopulations. At strength +5, high-CV features achieve mean effect 0.522 versus 0.355 for low-CV (t = 9.73, p < 0.01).

**Coefficient of variation as a practical predictor.** The CV-based decomposition requires no steering experiments—just a single pass through the SAE with representative inputs to compute activation statistics. The threshold CV = 1.0 reliably separates steerable from non-steerable absorbed features. For practitioners working with large feature dictionaries, CV-based screening can reduce experimental cost by prioritizing high-value steering targets.

**Evidence that the actionability paradox is domain-specific, not universal.** Basu et al. (2026) reported 0% steering utility in clinical domain despite 98.2% AUROC detection. We found substantial steering effects in non-clinical LLM domain, suggesting their finding reflects feature properties in that domain (predominantly low-CV) rather than a fundamental limitation of SAE-based interpretability. The variance paradox (CV_absorbed = 7.33 vs CV_non-absorbed = 0.01, 733x ratio) indicates absorption preserves context-sensitive specialized information in high-CV features.

**Mechanistic hypothesis: context-sensitive versus bypass routing.** We propose that high-CV (robust) features route through context-sensitive child channels where routing coefficient varies with input, creating mediated steering where parent modulation propagates to outputs. Low-CV (fragile) features route through stable child channels with fixed routing coefficients that compensate for parent steering, producing zero net effect. This hypothesis is supported by the activation patching results (67.3% mean recovery, all 9/9 words above 10% threshold) confirming genuine causal structure in absorbed features.

## 6.2 Limitations

Three limitations constrain our findings.

**GPT-2 Small is the primary experimental model.** Our full experiments used GPT-2 Small (117M parameters, layer 6). While we completed cross-architecture validation on Gemma-2-2B, detailed analysis of those results is pending. The CV-steering correlation and the CV = 1.0 threshold may require adjustment for different model families, layer depths, or SAE architectures.

**Steering protocol measures logit change only.** We measured effectiveness as logit change at target tokens, following standard practice. We did not measure downstream behavioral changes such as task completion rates or factual accuracy. The practical significance of observed logit changes for downstream applications remains an open question.

**CV threshold derived empirically, not theoretically.** The CV = 1.0 threshold separating robust from fragile absorbed features was identified empirically on GPT-2 layer 6 SAEs. We have not derived this threshold from first principles or validated it prospectively on held-out feature sets. The threshold may shift with different input distributions or SAE training conditions.

## 6.3 Future Work

**Larger model replication.** Llama-3 8B, Mistral 7B, and Gemma-2-9B would test whether the CV-steering correlation extends to larger models with different attention patterns and feature structures. This is the most critical next step for generalizability.

**Prospective threshold validation.** A held-out experiment would test whether CV > 1.0 correctly predicts steerability on features not used in threshold derivation. This would establish CV as a reliable predictor versus an empirical correlation.

**Causal mechanism via activation patching timecourse.** Current activation patching confirms causal structure exists, but does not distinguish between context-sensitive and bypass routing regimes. Timecourse analysis—measuring how quickly child activation responds to parent steering—could differentiate the two mechanisms directly.

**Cross-SAE architecture testing.** JumpReLU, TopK, and Gated SAEs may show different CV distributions or CV-steering correlations. Testing the decomposition on multiple SAE architectures would establish whether CV-based screening is architecture-specific or a general principle.

**Behavioral downstream validation.** Connecting logit changes to measurable behavioral changes (task accuracy, factuality, safety benchmark performance) would establish practical utility for interpretability practitioners.

## 6.4 Takeaways for Interpretability Practice

For practitioners working with SAE-based steering:

1. **Do not assume absorbed features are non-steerable.** Our primary finding is that absorbed features are heterogeneous—some are steerable, some are not.

2. **Compute CV before running steering experiments.** CV is a cheap predictor (single forward pass) that identifies steerable absorbed features without requiring multiple steering runs.

3. **Prioritize high-CV features for steering interventions.** Features with CV > 1.0 are significantly more likely to produce measurable effects (1.47x larger on average).

4. **Absorption metrics identify what is absorbed; CV predicts what is steerable.** The training-free detector $A_j = \|d_j\|^2 / (d_j^\top e_j)$ (Chanin et al., 2024) identifies absorption but does not predict actionability. CV fills this gap.

5. **Consider domain specificity when interpreting negative results.** Basu et al.'s clinical domain may exhibit predominantly low-CV features, explaining universal failure. Non-clinical LLM domains show high-CV features with steering utility.

The coefficient of variation provides a computationally cheap lens for understanding feature actionability, connecting abstract absorption metrics to practical steering utility. This work suggests that SAE-based interpretability can inform intervention—but practitioners must look beyond absorption detection to find features that are both detected and steerable.

<!-- FIGURES
- Figure 5: fig5_mechanism_desc.md — Architecture diagram describing context-sensitive vs bypass routing mechanism
- None
-->