# 8. Conclusion

## 8.1 Summary

We established the first connection between Rozell et al.'s Locally Competitive Algorithm (LCA) and feature absorption in sparse autoencoders. For tied-weight SAEs, the decoder correlation matrix $G = W_{\text{dec}}^T W_{\text{dec}}$ is exactly the LCA inhibition matrix. For standard untied SAEs, decoder correlations encode competitive suppression relationships between latents. This structural correspondence provides a mechanistic explanation for absorption: when a child feature fires strongly, it inhibits correlated parent features via decoder correlations, causing recall loss without affecting precision.

We constructed a local inhibition graph from decoder correlations and validated its predictive power against known absorption pairs. The graph is training-free, scalable to million-latent SAEs, and computed from pretrained weights in seconds. We also proposed homeostatic rebalancing---a single-pass post-hoc correction inspired by biological homeostatic plasticity---that restores parent firing by compensating for competitive suppression.

The competitive suppression framework explains all key findings from our prior empirical work. Precision is invariant because inhibition suppresses true positives but does not cause false positives. Recall varies because inhibition strength differs across feature pairs. Delta-corrected steering reveals absorption's effect at layer 8 because baseline subtraction isolates the unique information lost to inhibition. Feature U (24.2% absorption) still steers with 100% success because decoder directions are preserved; only encoder activations are suppressed.

## 8.2 Contributions

Our work makes five contributions:

1. **First connection between LCA lateral inhibition and SAE absorption.** We show that $W_{\text{dec}}^T W_{\text{dec}}$ is exactly the LCA inhibition matrix for tied-weight SAEs, providing an exact structural correspondence that has not been articulated in the SAE literature.

2. **First local inhibition graph for SAE diagnostics.** The graph is training-free, scalable to million-latent SAEs, and computed from pretrained weights. It predicts known absorption pairs with enrichment over chance and identifies at-risk features before running absorption metrics.

3. **Mechanistic explanation for precision--recall asymmetry.** Competitive suppression explains why precision is invariant while recall varies---a pattern observed in our prior experiments that lacked theoretical grounding.

4. **First training-free post-hoc repair for absorption.** Homeostatic rebalancing operates on pretrained SAEs with a single forward-pass correction, restoring parent firing while constraining reconstruction error increase to less than 5%.

5. **Integration of prior empirical findings into a unified framework.** The inhibition framework explains precision invariance, recall variation, layer-dependent effects, delta-corrected steering significance, and steering robustness under absorption as consequences of a single mechanism.

## 8.3 Closing Thought

Feature absorption has been identified, measured, standardized, and targeted by architectural innovations. Yet until now, the field has lacked a mechanistic theory that explains why absorption happens or identifies which features are at risk. The LCA--SAE correspondence fills this gap: absorption is not a mysterious pathology but competitive suppression, predictable from decoder correlations, and repairable with homeostatic rebalancing.

The implications extend beyond absorption. If decoder correlations encode competitive dynamics, then any SAE analysis that ignores these correlations---feature selection, steering vector design, circuit discovery---may miss a fundamental structural property of the representation. The local inhibition graph provides a lens for viewing SAEs not as collections of independent features but as networks of competing latents, with competitive relationships written directly into the decoder weights.

Whether this framework generalizes to larger models, semantic hierarchies, and alternative SAE architectures remains to be tested. We hope the theoretical tools and empirical methodology we provide enable the community to answer these questions.

<!-- FIGURES
- None
-->
