# 7. Conclusion

This paper presents the first connection between the Locally Competitive Algorithm from neuroscience and feature absorption in sparse autoencoders. We show that the SAE decoder correlation matrix $G = W_{\text{dec}}^T W_{\text{dec}}$ is exactly the LCA inhibition matrix for tied-weight SAEs, providing a mechanistic explanation for absorption as competitive suppression.

The empirical validation yields a nuanced picture. The primary predictive hypothesis (H6: graph edges predict absorption pairs) is falsified: precision@20 = 0.0, with no enrichment over chance ($p = 1.0$, Fisher exact test). Graph statistics similarly fail to predict at-risk features (H8: $r = +0.12$, $p = 0.55$). These null results indicate that the local inhibition graph with fixed $k = 20$ is too coarse to capture absorption-specific relationships.

However, the mechanistic framework is strongly supported. The precision-recall asymmetry (H7) matches the competitive suppression prediction exactly: precision standard deviation is 0.016 at $k_{\text{probe}} = 20$ (layer 8), while recall standard deviation is 0.167---more than 10x larger. Twenty-five of 26 first-letter features achieve precision = 1.0, while recall ranges from 0.21 to 0.49. Layer-dependent graph structure (H9) shows a descriptive trend: mean edge weight increases monotonically from 0.312 (layer 0) to 0.384 (layer 8), aligning with the prior finding that layer 8 exhibits the strongest absorption-steering correlation.

The inhibition framework provides a unified explanation for all key findings from prior experiments: precision invariance (inhibition suppresses true positives, not selectivity), recall loss (inhibition reduces parent activation when child fires), layer-dependent effects (deeper layers have stronger hierarchical structure = stronger inhibition), steering robustness (decoder direction preserved; only encoder activation suppressed), and the necessity of delta-corrected metrics (baseline subtraction isolates the signal-specific component that competitive suppression removes from encoder activation).

The practical contributions are two-fold. First, the framework transforms absorption from a mysterious pathology into a predictable consequence of competitive suppression, enabling practitioners to reason about at-risk features using decoder correlations rather than expensive pairwise absorption metrics. Second, homeostatic rebalancing provides a theoretical foundation for training-free repair, though its empirical validation awaits improved graph construction.

The central message is that absorption is not a critical failure mode for SAE interpretability. Competitive suppression reduces coverage (recall) but not selectivity (precision), and the decoder direction---the component used for steering and circuit analysis---remains intact. The LCA connection provides the theoretical grounding that the field needs to move from detecting absorption to understanding it.

Future work should test larger $k$ values, adaptive neighborhood sizes, and context-dependent edge weighting to improve graph-based prediction. Cross-architecture validation on JumpReLU, Gated, and TopK SAEs, and cross-model validation on Gemma-2-2B and Llama-3.1-8B, will test the generality of the competitive suppression mechanism. Semantic hierarchy features---animal $\rightarrow$ dog $\rightarrow$ poodle---will test whether the framework extends beyond first-letter tasks to deeper, more natural hierarchies.

<!-- FIGURES
- None
-->
